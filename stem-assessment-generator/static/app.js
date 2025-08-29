// STEM Assessment Generator Frontend JavaScript

class AssessmentGenerator {
    constructor() {
        this.taskId = null;
        this.uploadedFile = null;
        this.isProcessing = false;
        this.lastAssessment = null; // Store for download functionality
        this.chartObserver = null; // For intersection observer
        this.charts = new Map(); // Store chart instances for cleanup

        this.initializeElements();
        this.bindEvents();
        this.checkHealth();
        this.setupPerformanceObserver();
    }

    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadIcon = document.getElementById('uploadIcon');
        this.uploadText = document.getElementById('uploadText');
        this.uploadSubtext = document.getElementById('uploadSubtext');

        // Progress elements
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.progressPercent = document.getElementById('progressPercent');

        // Configuration elements
        this.configSection = document.getElementById('configSection');
        this.numQuestions = document.getElementById('numQuestions');
        this.difficulty = document.getElementById('difficulty');
        this.generateBtn = document.getElementById('generateBtn');

        // Section elements
        this.uploadSection = document.getElementById('uploadSection');
        this.loadingSection = document.getElementById('loadingSection');
        this.loadingTitle = document.getElementById('loadingTitle');
        this.loadingMessage = document.getElementById('loadingMessage');
        this.resultsSection = document.getElementById('resultsSection');
        this.questionsContainer = document.getElementById('questionsContainer');

        // Alert elements
        this.errorAlert = document.getElementById('errorAlert');
        this.errorMessage = document.getElementById('errorMessage');
        this.infoAlert = document.getElementById('infoAlert');
        this.infoMessage = document.getElementById('infoMessage');
        this.successAlert = document.getElementById('successAlert');
        this.successMessage = document.getElementById('successMessage');

        // Analytics elements
        this.analyticsSection = document.getElementById('analyticsSection');
        this.showAnalyticsBtn = document.getElementById('showAnalyticsBtn');

        // Chart variables
        this.difficultyChart = null;
        this.qualityChart = null;
    }

    bindEvents() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Generate button
        this.generateBtn.addEventListener('click', () => this.generateQuestions());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => this.handleBeforeUnload());

        // Performance: Debounced scroll events
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            if (scrollTimeout) clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                this.handleOptimizedScroll();
            }, 100);
        });
    }

    handleOptimizedScroll() {
        // Handle scroll-related performance optimizations
        // This is called at most once every 100ms
        if (this.resultsSection && !this.resultsSection.classList.contains('hidden')) {
            // Only perform scroll-related calculations when results are visible
            const rect = this.resultsSection.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
            
            if (isVisible) {
                // Results section is visible, could lazy load more content here
                // For now, just ensure smooth rendering
                requestAnimationFrame(() => {
                    // Any additional scroll-based optimizations
                });
            }
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();

            if (!health.openai_configured) {
                this.showError('OpenAI API key not configured. Some features may not work.');
            }
        } catch (error) {
            console.warn('Health check failed:', error);
        }
    }

    // =============================================================================
    // File Upload Handling
    // =============================================================================

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        this.uploadedFile = file;
        this.updateUploadArea(file);

        // Start upload
        await this.uploadFile(file);
    }

    validateFile(file) {
        // Check file type
        if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
            this.showError('Please select a PDF file.');
            return false;
        }

        // Check file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            this.showError(`File size exceeds 10MB limit. Selected file is ${(file.size / 1024 / 1024).toFixed(1)}MB.`);
            return false;
        }

        if (file.size === 0) {
            this.showError('File is empty. Please select a valid PDF file.');
            return false;
        }

        return true;
    }

    updateUploadArea(file) {
        this.uploadIcon.textContent = '✅';
        this.uploadText.textContent = `Selected: ${file.name}`;
        this.uploadSubtext.textContent = `Size: ${(file.size / 1024 / 1024).toFixed(1)}MB`;
        this.uploadArea.classList.add('border-green-500', 'bg-green-50');
    }

    resetUploadArea() {
        this.uploadIcon.textContent = '📄';
        this.uploadText.textContent = 'Drop your PDF here or click to browse';
        this.uploadSubtext.textContent = 'Supports PDF files up to 10MB';
        this.uploadArea.classList.remove('border-green-500', 'bg-green-50');
        this.fileInput.value = '';
    }

    // =============================================================================
    // File Upload API
    // =============================================================================

    async uploadFile(file) {
        try {
            this.showProgress(0, 'Uploading file...');
            this.isProcessing = true;

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            this.taskId = result.task_id;

            this.showProgress(20, 'Upload complete, processing...');
            this.showSuccess(`File uploaded successfully! Task ID: ${this.taskId.slice(0, 8)}...`);

            // Start monitoring the processing
            await this.monitorProcessing();

        } catch (error) {
            this.showError(`Upload failed: ${error.message}`);
            this.hideProgress();
            this.isProcessing = false;
        }
    }

    async monitorProcessing() {
        if (!this.taskId) return;

        try {
            const response = await fetch(`/api/status/${this.taskId}`);
            const status = await response.json();

            this.updateProgressFromStatus(status);

            if (status.status === 'ready') {
                this.showProgress(100, 'Processing complete!');
                setTimeout(() => {
                    this.hideProgress();
                    this.showConfigSection();
                    this.isProcessing = false;
                }, 1000);
            } else if (status.status === 'error') {
                throw new Error(status.error_details || 'Processing failed');
            } else {
                // Continue monitoring
                setTimeout(() => this.monitorProcessing(), 2000);
            }

        } catch (error) {
            this.showError(`Processing failed: ${error.message}`);
            this.hideProgress();
            this.isProcessing = false;
        }
    }

    updateProgressFromStatus(status) {
        const progress = (status.progress || 0) * 100;
        const statusMessages = {
            'uploading': 'Uploading file...',
            'processing': 'Processing PDF and building index...',
            'ready': 'Ready for question generation!',
            'error': 'Processing failed'
        };

        this.showProgress(progress, statusMessages[status.status] || status.message);
    }

    // =============================================================================
    // Question Generation
    // =============================================================================

    async generateQuestions() {
        if (!this.taskId) {
            this.showError('No document uploaded. Please upload a PDF first.');
            return;
        }

        if (this.isProcessing) {
            this.showError('Please wait for current processing to complete.');
            return;
        }

        try {
            this.isProcessing = true;
            this.showLoading('Generating questions...', 'AI is analyzing your document and creating assessment questions. This may take 30-60 seconds.');

            const request = {
                task_id: this.taskId,
                num_questions: parseInt(this.numQuestions.value),
                difficulty_level: this.difficulty.value
            };

            // Try enhanced endpoint first, fallback to standard
            let response;
            try {
                response = await fetch('/api/generate-with-progress', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(request)
                });

                if (response.ok && response.headers.get('content-type')?.includes('text/event-stream')) {
                    // Handle streaming response
                    await this.handleStreamingGeneration(response);
                    return;
                }
            } catch (streamError) {
                console.log('Streaming generation failed, falling back to standard generation:', streamError);
            }

            // Fallback to standard generation
            response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Question generation failed');
            }

            const assessment = await response.json();
            this.lastAssessment = assessment; // Store for download
            
            // Calculate and display analytics
            this.calculateAnalytics(assessment);
            
            this.displayResults(assessment);
            this.showSuccess(`Generated ${assessment.questions.length} questions successfully!`);

        } catch (error) {
            console.error('Generation error:', error);
            this.showEnhancedError(error);
            this.hideLoading();
        } finally {
            this.isProcessing = false;
        }
    }

    async handleStreamingGeneration(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let lastProgress = null;

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'progress') {
                                lastProgress = data;
                                this.updateProgressFromStream(data);
                            } else if (data.type === 'complete') {
                                this.lastAssessment = data.assessment;
                                this.displayResults(data.assessment);
                                this.showSuccess(`Generated ${data.assessment.questions.length} questions successfully!`);
                                return;
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (parseError) {
                            console.warn('Failed to parse SSE data:', line, parseError);
                        }
                    }
                }
            }
        } catch (error) {
            throw new Error(`Streaming generation failed: ${error.message}`);
        }
    }

    updateProgressFromStream(progressData) {
        const { questions_generated, total_requested, current_batch, status } = progressData;
        const percent = Math.min((questions_generated / total_requested) * 100, 95);

        this.showProgress(
            percent,
            `Generating question ${questions_generated + 1} of ${total_requested} (Batch ${current_batch})...`
        );
    }

    showEnhancedError(error) {
        let errorMessage = `Question generation failed: ${error.message}`;
        let suggestions = [];

        // Add specific suggestions based on error type
        if (error.message.includes('timeout')) {
            suggestions.push('Try reducing the number of questions');
            suggestions.push('Check your internet connection');
        } else if (error.message.includes('OpenAI')) {
            suggestions.push('Check if OpenAI API key is configured');
            suggestions.push('Try again in a few minutes');
        } else if (error.message.includes('chunk') || error.message.includes('content')) {
            suggestions.push('Try uploading a different PDF file');
            suggestions.push('Ensure the PDF contains readable text');
        }

        if (suggestions.length > 0) {
            errorMessage += '\n\nSuggestions:\n• ' + suggestions.join('\n• ');
        }

        this.showError(errorMessage);
    }

    // =============================================================================
    // UI State Management
    // =============================================================================

    showProgress(percent, message) {
        this.progressContainer.classList.remove('hidden');
        this.progressBar.style.width = `${percent}%`;
        this.progressText.textContent = message;
        this.progressPercent.textContent = `${Math.round(percent)}%`;
    }

    hideProgress() {
        this.progressContainer.classList.add('hidden');
    }

    showConfigSection() {
        this.configSection.classList.remove('hidden');
        this.generateBtn.classList.remove('hidden');
    }

    showLoading(title, message) {
        this.loadingTitle.textContent = title;
        this.loadingMessage.textContent = message;
        this.uploadSection.classList.add('hidden');
        this.loadingSection.classList.remove('hidden');
        this.resultsSection.classList.add('hidden');
    }

    hideLoading() {
        this.loadingSection.classList.add('hidden');
        this.uploadSection.classList.remove('hidden');
    }

    displayResults(assessment) {
        // Use DocumentFragment for batch DOM updates (performance optimization)
        const fragment = document.createDocumentFragment();

        // Create results summary
        const summaryCard = this.createResultsSummary(assessment);
        fragment.appendChild(summaryCard);

        // Create individual question cards using batch processing
        const batchSize = 5; // Process 5 questions at a time
        const questions = assessment.questions;
        
        const processBatch = (startIndex) => {
            const endIndex = Math.min(startIndex + batchSize, questions.length);
            
            for (let i = startIndex; i < endIndex; i++) {
                const questionCard = this.createQuestionCard(questions[i], i + 1);
                fragment.appendChild(questionCard);
            }
            
            if (endIndex < questions.length) {
                // Process next batch after a short delay to prevent UI blocking
                setTimeout(() => processBatch(endIndex), 0);
            } else {
                // All questions processed, update DOM once
                this.questionsContainer.innerHTML = '';
                this.questionsContainer.appendChild(fragment);
                
                this.uploadSection.classList.add('hidden');
                this.loadingSection.classList.add('hidden');
                this.resultsSection.classList.remove('hidden');
                
                // Scroll to results
                this.resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        };
        
        // Start processing the first batch
        processBatch(0);

        // Store assessment data for analytics
        this.currentAssessment = assessment;
    }

    showAnalytics() {
        if (!this.currentAssessment) {
            this.showError('No assessment data available for analytics');
            return;
        }

        // Calculate analytics
        this.calculateAnalytics(this.currentAssessment);
        
        // Show analytics section
        this.analyticsSection.classList.remove('hidden');
        
        // Hide the show analytics button
        this.showAnalyticsBtn.style.display = 'none';
        
        // Scroll to analytics
        this.analyticsSection.scrollIntoView({ behavior: 'smooth' });
    }

    calculateAnalytics(assessment) {
        try {
            const questions = assessment.questions || [];
            const totalQuestions = questions.length;
            
            if (totalQuestions === 0) {
                console.warn('No questions available for analytics calculation');
                return;
            }
            
            // Calculate quality metrics accurately
            let totalQualityScore = 0;
            let strongDistractors = 0;
            const difficultyCount = { EASY: 0, MEDIUM: 0, HARD: 0 };
            const concepts = new Set();
            const qualityScores = [];
            let totalDistractors = 0;
            
            questions.forEach((q, index) => {
                // Calculate quality score (0-1 scale) based on actual content
                let questionQuality = 0.3; // Base quality score
                
                // Check for question text quality
                if (q.question_text && q.question_text.trim().length > 20) {
                    questionQuality += 0.15;
                }
                
                // Check for basic explanation
                if (q.explanation && q.explanation.trim().length > 10) {
                    questionQuality += 0.15;
                }
                
                // Check for proper options structure
                const options = q.options || {};
                const optionCount = Object.keys(options).length;
                if (optionCount === 4) {
                    questionQuality += 0.1;
                }

                // Check for detailed option explanations and misconception analysis
                let hasDetailedExplanations = false;
                let misconceptionCount = 0;
                
                Object.entries(options).forEach(([key, option]) => {
                    if (typeof option === 'object') {
                        totalDistractors++;
                        
                        if (option.explanation) {
                            hasDetailedExplanations = true;
                            
                            // Count as strong distractor if it has misconception details
                            if (!option.is_correct) {
                                if (option.explanation.misconception_name || 
                                    option.explanation.why_students_think_this ||
                                    option.explanation.why_its_wrong) {
                                    strongDistractors++;
                                    misconceptionCount++;
                                }
                            }
                        }
                    } else if (typeof option === 'string' && option.length > 15) {
                        // Legacy format - count longer options as potentially strong
                        totalDistractors++;
                        if (!option.is_correct && option.length > 25) {
                            strongDistractors++;
                        }
                    }
                });

                // Quality bonus for detailed explanations
                if (hasDetailedExplanations) {
                    questionQuality += 0.2;
                }
                
                // Quality bonus for multiple misconceptions
                if (misconceptionCount >= 2) {
                    questionQuality += 0.1;
                }
                
                // Ensure quality doesn't exceed 1.0
                questionQuality = Math.min(1.0, questionQuality);
                
                totalQualityScore += questionQuality;
                qualityScores.push(questionQuality);
                
                // Count difficulty distribution accurately
                let difficulty = 'MEDIUM'; // Default
                if (q.difficulty) {
                    if (typeof q.difficulty === 'string') {
                        difficulty = q.difficulty.toUpperCase();
                    } else if (q.difficulty.value) {
                        difficulty = q.difficulty.value.toUpperCase();
                    }
                }
                
                if (difficultyCount.hasOwnProperty(difficulty)) {
                    difficultyCount[difficulty]++;
                } else {
                    difficultyCount.MEDIUM++;
                }
                
                // Collect concepts more accurately
                if (q.concept_tested) {
                    concepts.add(q.concept_tested);
                }
                if (q.metadata && q.metadata.concept) {
                    concepts.add(q.metadata.concept);
                }
                if (q.metadata && q.metadata.topic) {
                    concepts.add(q.metadata.topic);
                }
                
                // Extract concepts from question text
                const questionWords = q.question_text.toLowerCase();
                const conceptKeywords = ['blockchain', 'consensus', 'cryptocurrency', 'mining', 'hash', 'block', 'transaction', 'distributed', 'decentralized'];
                conceptKeywords.forEach(keyword => {
                    if (questionWords.includes(keyword)) {
                        concepts.add(keyword.charAt(0).toUpperCase() + keyword.slice(1));
                    }
                });
            });
            
            // Calculate averages and percentages
            const avgQualityScore = (totalQualityScore / totalQuestions).toFixed(2);
            const strongDistractorPercentage = totalDistractors > 0 ? 
                Math.round((strongDistractors / totalDistractors) * 100) : 0;
            
            // Store generation time from assessment or calculate
            let generationTime = '--';
            if (assessment.processing_time) {
                generationTime = Math.round(assessment.processing_time);
            } else if (assessment.statistics && assessment.statistics.generation_time) {
                generationTime = Math.round(assessment.statistics.generation_time);
            } else if (this.lastGenerationTime) {
                generationTime = this.lastGenerationTime;
            }
            
            // Update analytics display with accurate data
            document.getElementById('avgQualityScore').textContent = avgQualityScore;
            document.getElementById('strongDistractors').textContent = `${strongDistractorPercentage}%`;
            document.getElementById('conceptsCovered').textContent = concepts.size;
            document.getElementById('generationTime').textContent = generationTime;
            
            // Create charts with accurate data
            this.createDifficultyChart(difficultyCount);
            this.createQualityChart(qualityScores);
            
            // Show low quality questions
            this.showLowQualityQuestions(questions, qualityScores);
            
            console.log('Analytics calculated:', {
                avgQuality: avgQualityScore,
                strongDistractors: strongDistractorPercentage,
                concepts: concepts.size,
                difficulties: difficultyCount
            });
            
        } catch (error) {
            console.error('Analytics calculation error:', error);
            this.showError('Failed to calculate analytics: ' + error.message);
        }
    }

    createDifficultyChart(difficultyCount) {
        const ctx = document.getElementById('difficultyChart');
        if (!ctx) return;
        
        // Set canvas size to prevent expansion
        ctx.style.width = '300px';
        ctx.style.height = '200px';
        
        // Destroy existing chart if it exists
        if (this.difficultyChart) {
            this.difficultyChart.destroy();
        }
        
        this.difficultyChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Easy', 'Medium', 'Hard'],
                datasets: [{
                    data: [difficultyCount.EASY || 0, difficultyCount.MEDIUM || 0, difficultyCount.HARD || 0],
                    backgroundColor: ['#10B981', '#F59E0B', '#EF4444'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 10,
                            font: {
                                size: 11
                            }
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }

    createQualityChart(qualityScores) {
        const ctx = document.getElementById('qualityChart');
        if (!ctx) return;
        
        // Set canvas size to prevent expansion
        ctx.style.width = '300px';
        ctx.style.height = '200px';
        
        // Destroy existing chart if it exists
        if (this.qualityChart) {
            this.qualityChart.destroy();
        }

        // Create quality distribution (binned data)
        const qualityBins = { 'Low (0-0.5)': 0, 'Medium (0.5-0.8)': 0, 'High (0.8-1.0)': 0 };
        
        qualityScores.forEach(score => {
            if (score <= 0.5) {
                qualityBins['Low (0-0.5)']++;
            } else if (score <= 0.8) {
                qualityBins['Medium (0.5-0.8)']++;
            } else {
                qualityBins['High (0.8-1.0)']++;
            }
        });
        
        this.qualityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(qualityBins),
                datasets: [{
                    label: 'Number of Questions',
                    data: Object.values(qualityBins),
                    backgroundColor: ['#EF4444', '#F59E0B', '#10B981'],
                    borderColor: ['#DC2626', '#D97706', '#059669'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: Math.max(5, Math.max(...Object.values(qualityBins)) + 1),
                        ticks: {
                            stepSize: 1,
                            font: {
                                size: 11
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: 10
                            },
                            maxRotation: 45
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }

    showLowQualityQuestions(questions, qualityScores) {
        const lowQualityContainer = document.getElementById('lowQualityQuestions');
        lowQualityContainer.innerHTML = '';
        
        const lowQualityQuestions = questions
            .map((q, index) => ({ question: q, score: qualityScores[index], index: index + 1 }))
            .filter(item => item.score < 0.7)
            .sort((a, b) => a.score - b.score);
        
        if (lowQualityQuestions.length === 0) {
            lowQualityContainer.innerHTML = '<p class="text-green-600 text-sm">✅ All questions meet quality standards!</p>';
            return;
        }
        
        lowQualityQuestions.forEach(item => {
            const questionElement = document.createElement('div');
            questionElement.className = 'p-3 bg-yellow-50 border border-yellow-200 rounded-lg';
            questionElement.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <span class="font-medium text-yellow-800">Question ${item.index}</span>
                        <span class="ml-2 text-sm text-yellow-600">Quality: ${(item.score * 100).toFixed(0)}%</span>
                        <p class="text-sm text-gray-700 mt-1">${item.question.question_text?.substring(0, 100)}...</p>
                    </div>
                    <button onclick="assessmentGenerator.regenerateQuestion(${item.index - 1})" 
                            class="px-2 py-1 bg-yellow-500 hover:bg-yellow-600 text-white text-xs rounded">
                        Regenerate
                    </button>
                </div>
            `;
            lowQualityContainer.appendChild(questionElement);
        });
    }

    async regenerateQuestion(questionIndex) {
        if (!this.taskId) return;

        try {
            this.showInfo(`Regenerating question ${questionIndex + 1}...`);
            
            const response = await fetch(`/api/regenerate/${this.taskId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question_indices: [questionIndex],
                    regenerate_all_below_threshold: false
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.showSuccess(`Successfully regenerated question ${questionIndex + 1}`);
                // Refresh the assessment display
                await this.checkTask();
                this.loadAnalytics();
            } else {
                const error = await response.json();
                this.showError(`Failed to regenerate question: ${error.detail}`);
            }
        } catch (error) {
            console.error('Question regeneration error:', error);
            this.showError(`Failed to regenerate question ${questionIndex + 1}`);
        }
    }



    // =============================================================================
    // Results Display and Analytics  
    // =============================================================================

    displayAnalyticsData(analytics) {
        try {
            // Update summary metrics safely
            const avgQualityElement = document.getElementById('avgQualityScore');
            const strongDistractorsElement = document.getElementById('strongDistractors');
            const conceptsCoveredElement = document.getElementById('conceptsCovered');
            const generationTimeElement = document.getElementById('generationTime');
            
            if (avgQualityElement) {
                avgQualityElement.textContent = analytics.summary.average_quality_score?.toFixed(1) + '%' || '--';
            }
            if (strongDistractorsElement) {
                strongDistractorsElement.textContent = analytics.summary.strong_distractors_count || '--';
            }
            if (conceptsCoveredElement) {
                conceptsCoveredElement.textContent = analytics.summary.concepts_covered?.length || '--';
            }
            if (generationTimeElement) {
                generationTimeElement.textContent = 
                    analytics.summary.generation_time ? analytics.summary.generation_time.toFixed(1) + 's' : '--';
            }

            // Setup chart containers for lazy loading
            setTimeout(() => {
                const difficultyContainer = document.getElementById('difficultyChart')?.parentElement;
                const qualityContainer = document.getElementById('qualityChart')?.parentElement;

                if (difficultyContainer && this.chartObserver) {
                    difficultyContainer.dataset.chartType = 'difficulty';
                    difficultyContainer.classList.add('chart-container');
                    this.chartObserver.observe(difficultyContainer);
                }

                if (qualityContainer && this.chartObserver) {
                    qualityContainer.dataset.chartType = 'quality';
                    qualityContainer.classList.add('chart-container');
                    this.chartObserver.observe(qualityContainer);
                }
            }, 500); // Delay chart rendering for better performance

        } catch (error) {
            console.error('Analytics display failed:', error);
        }
    }

    createResultsSummary(assessment) {
        const summaryCard = document.createElement('div');
        summaryCard.className = 'bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 shadow-lg border border-blue-200 mb-6';

        const stats = assessment.statistics || {};
        const processingTime = stats.generation_time || 'N/A';
        const totalQuestions = assessment.questions.length;

        // Calculate difficulty distribution
        const difficultyCount = { EASY: 0, MEDIUM: 0, HARD: 0 };
        assessment.questions.forEach(q => {
            if (q.difficulty) difficultyCount[q.difficulty.toUpperCase()]++;
        });

        summaryCard.innerHTML = `
            <div class="flex justify-between items-start mb-4">
                <h2 class="text-2xl font-bold text-gray-800">Assessment Generated Successfully</h2>
                <div class="flex flex-wrap gap-2">
                    <button onclick="assessmentGenerator.downloadAsJSON()" 
                            class="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm">
                        📥 JSON
                    </button>
                    <button onclick="assessmentGenerator.downloadAsPDF()" 
                            class="px-3 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors text-sm">
                        📄 Teacher PDF
                    </button>
                    <button onclick="assessmentGenerator.downloadStudentPDF()" 
                            class="px-3 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors text-sm">
                        📄 Student PDF
                    </button>
                    <button onclick="assessmentGenerator.downloadAsWord()" 
                            class="px-3 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors text-sm">
                        📝 Teacher Word
                    </button>
                    <button onclick="assessmentGenerator.downloadStudentWord()" 
                            class="px-3 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600 transition-colors text-sm">
                        📝 Student Word
                    </button>
                </div>
            </div>
            
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div class="text-center p-3 bg-white rounded-lg">
                    <div class="text-2xl font-bold text-blue-600">${totalQuestions}</div>
                    <div class="text-sm text-gray-600">Total Questions</div>
                </div>
                <div class="text-center p-3 bg-white rounded-lg">
                    <div class="text-2xl font-bold text-green-600">${difficultyCount.EASY}</div>
                    <div class="text-sm text-gray-600">Easy</div>
                </div>
                <div class="text-center p-3 bg-white rounded-lg">
                    <div class="text-2xl font-bold text-yellow-600">${difficultyCount.MEDIUM}</div>
                    <div class="text-sm text-gray-600">Medium</div>
                </div>
                <div class="text-center p-3 bg-white rounded-lg">
                    <div class="text-2xl font-bold text-red-600">${difficultyCount.HARD}</div>
                    <div class="text-sm text-gray-600">Hard</div>
                </div>
            </div>
            
            <div class="text-sm text-gray-600">
                <span class="font-semibold">Generation Time:</span> ${processingTime}s | 
                <span class="font-semibold">Source:</span> ${assessment.source_file || 'Unknown'} |
                <span class="font-semibold">Generated:</span> ${new Date().toLocaleString()}
            </div>
        `;

        return summaryCard;
    }

    createQuestionCard(question, questionNumber) {
        const card = document.createElement('div');
        card.className = 'question-card bg-white rounded-2xl p-6 shadow-lg border border-gray-100 mb-6';

        // Parse the new question format with detailed explanations
        let allOptions = [];
        let optionsObject = {};

        // Check if we have detailed options in metadata
        if (question.metadata?.detailed_options) {
            // New format with detailed explanations
            optionsObject = question.metadata.detailed_options;
            allOptions = Object.entries(question.metadata.detailed_options).map(([key, option]) => ({
                letter: key,
                text: option.text,
                isCorrect: option.is_correct,
                misconception: option.explanation?.misconception || null,
                explanation: option.explanation || {}
            }));
        } else if (question.options && typeof question.options === 'object') {
            // Standard format: {A: "text", B: "text", C: "text", D: "text"}
            optionsObject = question.options;
            allOptions = Object.entries(question.options).map(([key, value]) => ({
                letter: key,
                text: value,
                isCorrect: question.correct_position === key,
                misconception: question.misconceptions && question.misconceptions[key] !== 'none' ? question.misconceptions[key] : null,
                explanation: {}
            }));
        } else {
            // Legacy format: correct_answer + distractors
            const correctAnswer = {
                letter: 'A',
                text: question.correct_answer,
                isCorrect: true,
                misconception: null,
                explanation: {}
            };

            const distractors = (question.distractors || []).map((d, index) => ({
                letter: String.fromCharCode(66 + index), // B, C, D
                text: d.text || d,
                isCorrect: false,
                misconception: d.misconception_type || null,
                explanation: {}
            }));

            allOptions = [correctAnswer, ...distractors];

            // Shuffle options for legacy format
            this.shuffleArray(allOptions);

            // Reassign letters after shuffling
            allOptions.forEach((option, index) => {
                option.letter = String.fromCharCode(65 + index);
            });
        }

        const optionsHtml = allOptions.map((option) => {
            const misconceptionData = option.misconception ?
                `data-misconception="${option.misconception}"` : '';

            return `
                <div class="option option-hover p-4 rounded-lg border-2 border-gray-200 cursor-pointer transition-all duration-200 mb-3" 
                     data-correct="${option.isCorrect}" 
                     ${misconceptionData}
                     onclick="assessmentGenerator.selectOption(this)">
                    <div class="flex items-start">
                        <span class="font-bold text-gray-600 mr-3 mt-1">${option.letter}.</span>
                        <span class="flex-1">${option.text}</span>
                        <span class="misconception-indicator hidden text-xs text-orange-500 ml-2">⚠️</span>
                    </div>
                </div>
            `;
        }).join('');

        const difficultyColors = {
            'EASY': 'bg-green-100 text-green-800',
            'MEDIUM': 'bg-yellow-100 text-yellow-800',
            'HARD': 'bg-red-100 text-red-800',
            'easy': 'bg-green-100 text-green-800',
            'medium': 'bg-yellow-100 text-yellow-800',
            'hard': 'bg-red-100 text-red-800'
        };

        const questionDifficulty = question.difficulty || 'MEDIUM';
        const difficultyClass = difficultyColors[questionDifficulty] || difficultyColors['MEDIUM'];

        card.innerHTML = `
            <div class="flex justify-between items-start mb-4">
                <span class="bg-blue-500 text-white px-4 py-2 rounded-full text-sm font-semibold">
                    Question ${questionNumber}
                </span>
                <div class="flex items-center space-x-2">
                    <span class="px-3 py-1 rounded-full text-xs font-medium ${difficultyClass}">
                        ${questionDifficulty.toUpperCase()}
                    </span>
                    <button onclick="assessmentGenerator.copyQuestion(${questionNumber}, this)" 
                            class="copy-btn p-2 text-gray-500 hover:text-blue-500 rounded-lg hover:bg-blue-50" 
                            title="Copy question">
                        📋
                    </button>
                    <button onclick="assessmentGenerator.showAnswer(this)" 
                            class="show-answer-btn p-2 text-gray-500 hover:text-green-500 rounded-lg hover:bg-green-50" 
                            title="Show correct answer">
                        👁️
                    </button>
                </div>
            </div>
            
            <div class="question-text text-lg mb-6 leading-relaxed text-gray-800">
                ${question.question_text}
            </div>
            
            <div class="options mb-6">
                ${optionsHtml}
            </div>
            
            <div class="explanation hidden p-4 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                <h4 class="font-semibold text-blue-800 mb-2">Detailed Explanation:</h4>
                <div class="explanation-content">
                    ${this.renderDetailedExplanation(question)}
                </div>
            </div>
            
            ${question.metadata ? `
                <div class="metadata text-xs text-gray-500 mt-4 p-3 bg-gray-50 rounded-lg">
                    <strong>Source:</strong> ${question.metadata.concept || 'N/A'} | 
                    <strong>Chunk:</strong> ${question.metadata.source_chunk_id || 'N/A'}
                </div>
            ` : ''}
        `;

        return card;
    }

    renderDetailedExplanation(question) {
        // Handle both new format with detailed explanations and legacy format
        if (question.options && typeof question.options === 'object') {
            // New format with detailed explanations
            let explanationHtml = '';
            
            // Overall explanation if available
            if (question.overall_explanation) {
                explanationHtml += `
                    <div class="mb-6 p-4 bg-blue-100 rounded-lg border-l-4 border-blue-500">
                        <h5 class="font-semibold text-blue-800 mb-2">📚 Key Concept</h5>
                        <p class="text-blue-700 leading-relaxed">${question.overall_explanation}</p>
                    </div>
                `;
            }

            // Find correct answer first
            let correctOption = null;
            let incorrectOptions = [];
            
            Object.entries(question.options).forEach(([key, option]) => {
                if (option.is_correct) {
                    correctOption = { key, option };
                } else {
                    incorrectOptions.push({ key, option });
                }
            });

            // Show incorrect options first with misconceptions
            if (incorrectOptions.length > 0) {
                explanationHtml += `
                    <div class="mb-6">
                        <h5 class="font-semibold text-red-800 mb-4 flex items-center">
                            <span class="mr-2">🚫</span> Common Misconceptions Students Have
                        </h5>
                `;

                incorrectOptions.forEach(({ key, option }) => {
                    const explanation = option.explanation || {};
                    explanationHtml += `
                        <div class="mb-4 p-4 bg-red-50 rounded-lg border-l-4 border-red-400">
                            <h6 class="font-semibold text-red-800 mb-3">
                                Option ${key} – "${option.text}"
                            </h6>
                            <div class="space-y-3 text-sm text-red-700">
                                <div>
                                    <span class="font-medium">• Misconception:</span> 
                                    ${explanation.why_students_think_this || `Students often select this option due to ${explanation.misconception_name || 'surface-level understanding'}.`}
                                </div>
                                <div>
                                    <span class="font-medium">• Why Wrong:</span> 
                                    ${explanation.why_its_wrong || 'This option represents a common misunderstanding of the core concept.'}
                                </div>
                                ${explanation.correct_understanding ? `
                                    <div>
                                        <span class="font-medium">• Correct Understanding:</span> 
                                        ${explanation.correct_understanding}
                                    </div>
                                ` : ''}
                                ${explanation.remember_this ? `
                                    <div class="bg-red-100 p-2 rounded border border-red-200">
                                        <span class="font-medium">💡 Remember:</span> 
                                        ${explanation.remember_this}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                });

                explanationHtml += `</div>`;
            }

            // Show correct answer with detailed explanation
            if (correctOption) {
                const { key, option } = correctOption;
                const explanation = option.explanation || {};
                
                explanationHtml += `
                    <div class="mb-4 p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                        <h5 class="font-semibold text-green-800 mb-3 flex items-center">
                            <span class="mr-2">✅</span> Correct Answer (Option ${key})
                        </h5>
                        <div class="mb-3">
                            <div class="font-medium text-green-800 text-lg mb-2">"${option.text}"</div>
                        </div>
                        
                        <div class="space-y-3 text-sm text-green-700">
                            <div>
                                <span class="font-semibold">Why It's Correct:</span>
                                <div class="mt-1 leading-relaxed">
                                    ${explanation.why_correct || 'This option demonstrates the correct understanding of the fundamental concept being tested.'}
                                </div>
                            </div>
                            
                            ${explanation.key_points && explanation.key_points.length > 0 ? `
                                <div>
                                    <span class="font-semibold">Key Points:</span>
                                    <ul class="mt-1 ml-4 space-y-1">
                                        ${explanation.key_points.map(point => `<li class="list-disc">${point}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            
                            ${explanation.real_world_example ? `
                                <div>
                                    <span class="font-semibold">Real-World Application:</span>
                                    <div class="mt-1 leading-relaxed">${explanation.real_world_example}</div>
                                </div>
                            ` : ''}
                            
                            ${explanation.connects_to ? `
                                <div class="bg-green-100 p-3 rounded border border-green-200">
                                    <span class="font-semibold">🔗 Connects To:</span> ${explanation.connects_to}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }

            return explanationHtml || `
                <div class="p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-400">
                    <p class="text-yellow-700">
                        <span class="font-medium">Note:</span> Detailed explanations are being generated. 
                        Please regenerate questions to get comprehensive misconception analysis.
                    </p>
                </div>
            `;
        } else {
            // Legacy format - show basic explanation
            return `
                <div class="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                    <p class="text-blue-700">${question.explanation || 'No detailed explanation available.'}</p>
                    <div class="mt-3 p-3 bg-blue-100 rounded border border-blue-200">
                        <span class="font-medium text-blue-800">💡 Tip:</span> 
                        <span class="text-blue-700">Regenerate this question to get detailed misconception analysis for each option.</span>
                    </div>
                </div>
            `;
        }
    }

    renderOptionExplanation(explanation, isCorrect) {
        if (isCorrect) {
            return `
                <div class="text-green-700 text-sm space-y-1">
                    ${explanation.why_correct ? `<p><strong>Why correct:</strong> ${explanation.why_correct}</p>` : 
                      '<p><strong>Why correct:</strong> This option demonstrates the correct understanding of the concept.</p>'}
                    ${explanation.key_concept ? `<p><strong>Key concept:</strong> ${explanation.key_concept}</p>` : ''}
                    ${explanation.related_info ? `<p><strong>Related info:</strong> ${explanation.related_info}</p>` : ''}
                </div>
            `;
        } else {
            return `
                <div class="text-red-700 text-sm space-y-1">
                    ${explanation.misconception && explanation.misconception !== 'none' ? 
                      `<p><strong>Misconception:</strong> ${explanation.misconception}</p>` : 
                      '<p><strong>Common Error:</strong> This represents a typical misunderstanding.</p>'}
                    ${explanation.why_chosen ? `<p><strong>Why students choose this:</strong> ${explanation.why_chosen}</p>` : 
                      '<p><strong>Why students choose this:</strong> Students often select this due to partial understanding or confusion.</p>'}
                    ${explanation.correction ? `<p><strong>Correction:</strong> ${explanation.correction}</p>` : ''}
                    ${explanation.hint ? `<p><strong>Hint:</strong> ${explanation.hint}</p>` : 
                      '<p><strong>Tip:</strong> Review the key concepts to better understand why this option is incorrect.</p>'}
                </div>
            `;
        }
    }

    selectOption(optionElement) {
        const card = optionElement.closest('.question-card');
        const allOptions = card.querySelectorAll('.option');
        const explanation = card.querySelector('.explanation');

        // Remove previous selections
        allOptions.forEach(opt => {
            opt.classList.remove('border-green-500', 'bg-green-50', 'border-red-500', 'bg-red-50');
        });

        // Mark selected option
        const isCorrect = optionElement.dataset.correct === 'true';
        if (isCorrect) {
            optionElement.classList.add('border-green-500', 'bg-green-50');
        } else {
            optionElement.classList.add('border-red-500', 'bg-red-50');

            // Show correct answer
            allOptions.forEach(opt => {
                if (opt.dataset.correct === 'true') {
                    opt.classList.add('border-green-500', 'bg-green-50');
                }
            });
        }

        // Show misconception indicators after answer reveal
        allOptions.forEach(opt => {
            const indicator = opt.querySelector('.misconception-indicator');
            if (indicator && opt.dataset.misconception && opt.dataset.misconception !== 'undefined') {
                indicator.classList.remove('hidden');
                indicator.title = `Common misconception: ${opt.dataset.misconception}`;
            }
        });

        // Show explanation
        explanation.classList.remove('hidden');

        // Disable further clicks
        allOptions.forEach(opt => {
            opt.style.pointerEvents = 'none';
            opt.classList.remove('option-hover');
        });
    }

    showAnswer(buttonElement) {
        const card = buttonElement.closest('.question-card');
        const allOptions = card.querySelectorAll('.option');
        const explanation = card.querySelector('.explanation');

        // Highlight correct answer
        allOptions.forEach(opt => {
            if (opt.dataset.correct === 'true') {
                opt.classList.add('border-green-500', 'bg-green-50');
            }
        });

        // Show misconception indicators after answer reveal
        allOptions.forEach(opt => {
            const indicator = opt.querySelector('.misconception-indicator');
            if (indicator && opt.dataset.misconception && opt.dataset.misconception !== 'undefined') {
                indicator.classList.remove('hidden');
                indicator.title = `Common misconception: ${opt.dataset.misconception}`;
            }
        });

        // Show explanation
        explanation.classList.remove('hidden');

        // Update button
        buttonElement.innerHTML = '✅';
        buttonElement.classList.add('text-green-500');
        buttonElement.disabled = true;
    }

    async downloadAsJSON() {
        if (!this.lastAssessment) {
            this.showError('No assessment data available to download');
            return;
        }

        const dataStr = JSON.stringify(this.lastAssessment, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `assessment_${new Date().toISOString().split('T')[0]}.json`;
        link.click();

        this.showSuccess('Assessment downloaded as JSON');
    }

    async downloadAsPDF() {
        if (!this.taskId) {
            this.showError('No task available for PDF download');
            return;
        }

        try {
            // Show loading state
            const downloadBtn = document.querySelector('button[onclick="assessmentGenerator.downloadAsPDF()"]');
            const originalText = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '⏳ Generating PDF...';
            downloadBtn.disabled = true;

            // Request PDF generation
            const response = await fetch(`/api/export/${this.taskId}/pdf`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    teacher_version: true,
                    include_explanations: true
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to generate PDF: ${response.status}`);
            }

            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `assessment_${this.taskId}_${new Date().toISOString().split('T')[0]}.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);

            this.showSuccess('PDF downloaded successfully!');

        } catch (error) {
            console.error('PDF download failed:', error);
            this.showError(`PDF download failed: ${error.message}`);
        } finally {
            // Restore button state
            const downloadBtn = document.querySelector('button[onclick="assessmentGenerator.downloadAsPDF()"]');
            if (downloadBtn) {
                downloadBtn.innerHTML = originalText || '📄 Download PDF';
                downloadBtn.disabled = false;
            }
        }
    }

    async downloadStudentPDF() {
        if (!this.taskId) {
            this.showError('No task available for PDF download');
            return;
        }

        try {
            // Show loading state
            const downloadBtn = document.querySelector('button[onclick="assessmentGenerator.downloadStudentPDF()"]');
            const originalText = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '⏳ Generating PDF...';
            downloadBtn.disabled = true;

            // Request PDF generation (student version)
            const response = await fetch(`/api/export/${this.taskId}/pdf`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    teacher_version: false,
                    include_explanations: false
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to generate student PDF: ${response.status}`);
            }

            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `assessment_student_${this.taskId}_${new Date().toISOString().split('T')[0]}.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);

            this.showSuccess('Student PDF downloaded successfully!');

        } catch (error) {
            console.error('Student PDF download failed:', error);
            this.showError(`Student PDF download failed: ${error.message}`);
        } finally {
            // Restore button state
            const downloadBtn = document.querySelector('button[onclick="assessmentGenerator.downloadStudentPDF()"]');
            if (downloadBtn) {
                downloadBtn.innerHTML = originalText || '📄 Student Version';
                downloadBtn.disabled = false;
            }
        }
    }

    async downloadAsWord() {
        if (!this.taskId) {
            this.showError('No assessment available to download');
            return;
        }

        const downloadBtn = document.querySelector('button[onclick="assessmentGenerator.downloadAsWord()"]');
        const originalText = downloadBtn ? downloadBtn.innerHTML : null;

        try {
            // Update button state
            if (downloadBtn) {
                downloadBtn.innerHTML = '⏳ Generating Word...';
                downloadBtn.disabled = true;
            }

            this.showInfo('Generating Word document... This may take a moment.');

            const response = await fetch(`/api/export/${this.taskId}/docx?teacher_version=true`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Word generation failed with status ${response.status}`);
            }

            // Download the Word document
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `assessment_teacher_${this.taskId}_${new Date().toISOString().split('T')[0]}.docx`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);

            this.showSuccess('Teacher Word document downloaded successfully!');

        } catch (error) {
            console.error('Word download failed:', error);
            this.showError(`Word download failed: ${error.message}`);
        } finally {
            // Restore button state
            if (downloadBtn) {
                downloadBtn.innerHTML = originalText || '📝 Teacher Word';
                downloadBtn.disabled = false;
            }
        }
    }

    async downloadStudentWord() {
        if (!this.taskId) {
            this.showError('No assessment available to download');
            return;
        }

        const downloadBtn = document.querySelector('button[onclick="assessmentGenerator.downloadStudentWord()"]');
        const originalText = downloadBtn ? downloadBtn.innerHTML : null;

        try {
            // Update button state
            if (downloadBtn) {
                downloadBtn.innerHTML = '⏳ Generating Word...';
                downloadBtn.disabled = true;
            }

            this.showInfo('Generating student Word document... This may take a moment.');

            const response = await fetch(`/api/export/${this.taskId}/docx?teacher_version=false`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Word generation failed with status ${response.status}`);
            }

            // Download the Word document
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `assessment_student_${this.taskId}_${new Date().toISOString().split('T')[0]}.docx`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);

            this.showSuccess('Student Word document downloaded successfully!');

        } catch (error) {
            console.error('Student Word download failed:', error);
            this.showError(`Student Word download failed: ${error.message}`);
        } finally {
            // Restore button state
            if (downloadBtn) {
                downloadBtn.innerHTML = originalText || '📝 Student Word';
                downloadBtn.disabled = false;
            }
        }
    }

    async copyQuestion(questionNumber, buttonElement) {
        const card = buttonElement.closest('.question-card');
        const questionText = card.querySelector('.question-text').textContent;
        const options = Array.from(card.querySelectorAll('.option')).map((opt, index) => {
            const letter = String.fromCharCode(65 + index);
            const text = opt.textContent.trim().substring(2); // Remove "A. " prefix
            return `${letter}. ${text}`;
        });

        const copyText = `Question ${questionNumber}:\n${questionText}\n\n${options.join('\n')}`;

        try {
            await navigator.clipboard.writeText(copyText);

            // Visual feedback
            const originalText = buttonElement.textContent;
            buttonElement.textContent = '✅';
            buttonElement.classList.add('text-green-500');

            setTimeout(() => {
                buttonElement.textContent = originalText;
                buttonElement.classList.remove('text-green-500');
            }, 2000);

            this.showSuccess('Question copied to clipboard!');
        } catch (error) {
            this.showError('Failed to copy to clipboard');
        }
    }

    // =============================================================================
    // Alert Management
    // =============================================================================

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorAlert.classList.remove('hidden');

        // Auto-hide after 5 seconds
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        this.errorAlert.classList.add('hidden');
    }

    showInfo(message) {
        this.infoMessage.textContent = message;
        this.infoAlert.classList.remove('hidden');

        // Auto-hide after 10 seconds
        setTimeout(() => this.hideInfo(), 10000);
    }

    hideInfo() {
        this.infoAlert.classList.add('hidden');
    }

    showSuccess(message) {
        this.successMessage.textContent = message;
        this.successAlert.classList.remove('hidden');

        // Auto-hide after 3 seconds
        setTimeout(() => this.hideSuccess(), 3000);
    }

    hideSuccess() {
        this.successAlert.classList.add('hidden');
    }

    // =============================================================================
    // Utility Functions
    // =============================================================================

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    handleKeyboard(e) {
        // Escape to start over
        if (e.key === 'Escape') {
            if (!this.resultsSection.classList.contains('hidden')) {
                this.startOver();
            }
        }
    }

    // =============================================================================
    // Global Functions
    // =============================================================================

    startOver() {
        // Reset state
        this.taskId = null;
        this.uploadedFile = null;
        this.isProcessing = false;

        // Reset UI
        this.resetUploadArea();
        this.hideProgress();
        this.configSection.classList.add('hidden');
        this.generateBtn.classList.add('hidden');

        // Show upload section
        this.uploadSection.classList.remove('hidden');
        this.loadingSection.classList.add('hidden');
        this.resultsSection.classList.add('hidden');

        // Clear questions
        this.questionsContainer.innerHTML = '';

        // Hide alerts
        this.hideError();
        this.hideSuccess();

        // Scroll to top
        document.querySelector('header').scrollIntoView({ behavior: 'smooth' });
    }

    // =============================================================================
    // Export Functions
    // =============================================================================

    async exportQuestions() {
        if (!this.taskId) {
            this.showError('No questions available to export');
            return;
        }

        try {
            const format = document.getElementById('exportFormat').value;
            const includeExplanations = document.getElementById('includeExplanations').checked;
            const includeDifficulty = document.getElementById('includeDifficulty').checked;
            const randomizeOrder = document.getElementById('randomizeOrder').checked;
            const separateAnswerKey = document.getElementById('separateAnswerKey').checked;

            const response = await fetch(`/api/export/${this.taskId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    format_type: format,
                    teacher_version: !format.includes('student'),
                    include_explanations: includeExplanations,
                    include_difficulty: includeDifficulty,
                    randomize_order: randomizeOrder,
                    separate_answer_key: separateAnswerKey
                })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `assessment_${this.taskId}.${this.getFileExtension(format)}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

                this.showSuccess('Assessment exported successfully!');
            } else {
                const error = await response.json();
                this.showError(`Export failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Export error:', error);
            this.showError('Failed to export assessment');
        }
    }

    getFileExtension(format) {
        const extensions = {
            'pdf-teacher': 'pdf',
            'pdf-student': 'pdf',
            'json': 'json',
            'docx': 'docx',
            'txt': 'txt',
            'moodle': 'xml'
        };
        return extensions[format] || 'txt';
    }

    async previewExport() {
        if (!this.lastAssessment) {
            this.showError('No assessment data available for preview');
            return;
        }

        // Create a modal with preview content
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-4xl max-h-96 overflow-y-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">Export Preview</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700">✕</button>
                </div>
                <div class="space-y-4">
                    ${this.lastAssessment.questions.map((q, i) => `
                        <div class="border-b pb-4">
                            <h4 class="font-medium">Question ${i + 1}:</h4>
                            <p class="mb-2">${q.question_text}</p>
                            <div class="grid grid-cols-2 gap-2 text-sm">
                                ${Object.entries(q.options).map(([key, value]) => `
                                    <div class="${key === q.correct_position ? 'font-bold text-green-600' : ''}">${key}. ${value}</div>
                                `).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // =============================================================================
    // Cleanup and Performance Management
    // =============================================================================

    cleanup() {
        // Destroy all chart instances
        this.charts.forEach((chart, key) => {
            try {
                chart.destroy();
            } catch (error) {
                console.warn(`Failed to destroy chart ${key}:`, error);
            }
        });
        this.charts.clear();

        // Disconnect intersection observer
        if (this.chartObserver) {
            this.chartObserver.disconnect();
        }
    }

    // Call cleanup when page is about to unload
    handleBeforeUnload() {
        this.cleanup();
    }

    // =============================================================================
    // Analytics Functions
    // =============================================================================

    async loadAnalytics() {
        if (!this.taskId) return;

        try {
            const response = await fetch(`/api/analytics/${this.taskId}`);
            if (response.ok) {
                const analytics = await response.json();
                this.displayAnalytics(analytics);
            } else {
                console.warn('Analytics not available for this task');
            }
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }

    setupPerformanceObserver() {
        // Intersection Observer for lazy loading charts
        const observerOptions = {
            root: null,
            rootMargin: '50px',
            threshold: 0.1
        };

        this.chartObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    requestAnimationFrame(() => {
                        this.renderChart(entry.target);
                        this.chartObserver.unobserve(entry.target);
                    });
                }
            });
        }, observerOptions);
    }

    displayAnalytics(analytics) {
        // Update summary metrics
        document.getElementById('avgQualityScore').textContent =
            analytics.summary.average_quality_score?.toFixed(1) || '--';
        document.getElementById('strongDistractors').textContent =
            analytics.summary.strong_distractors_count || '--';
        document.getElementById('conceptsCovered').textContent =
            analytics.summary.concepts_covered?.length || '--';
        document.getElementById('generationTime').textContent =
            analytics.summary.generation_time?.toFixed(1) || '--';

        // Store analytics data for lazy chart rendering
        this.analyticsData = analytics;

        // Setup chart containers for lazy loading
        const difficultyContainer = document.getElementById('difficultyChart')?.parentElement;
        const qualityContainer = document.getElementById('qualityChart')?.parentElement;

        if (difficultyContainer) {
            difficultyContainer.dataset.chartType = 'difficulty';
            difficultyContainer.classList.add('chart-container');
            this.chartObserver.observe(difficultyContainer);
        }

        if (qualityContainer) {
            qualityContainer.dataset.chartType = 'quality';
            qualityContainer.classList.add('chart-container');
            this.chartObserver.observe(qualityContainer);
        }

        // Display low quality questions (lightweight)
        this.displayLowQualityQuestions(analytics.low_quality_questions || []);
    }

    renderChart(container) {
        const chartType = container.dataset.chartType;
        
        if (!this.analyticsData) return;

        // Show loading state
        const canvas = container.querySelector('canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            
            // Use setTimeout to prevent UI blocking
            setTimeout(() => {
                try {
                    if (chartType === 'difficulty') {
                        this.createOptimizedDifficultyChart(ctx, this.analyticsData.difficulty_distribution);
                    } else if (chartType === 'quality') {
                        this.createOptimizedQualityChart(ctx, this.analyticsData.quality_metrics);
                    }
                } catch (error) {
                    console.error('Chart rendering failed:', error);
                }
            }, 0);
        }
    }

    createOptimizedDifficultyChart(ctx, distribution) {
        // Clean up existing chart
        const existingChart = this.charts.get('difficulty');
        if (existingChart) {
            existingChart.destroy();
        }

        const chart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Easy', 'Medium', 'Hard'],
                datasets: [{
                    data: [
                        distribution?.EASY || 0,
                        distribution?.MEDIUM || 0,
                        distribution?.HARD || 0
                    ],
                    backgroundColor: ['#10B981', '#F59E0B', '#EF4444']
                }]
            },
            options: {
                responsive: false, // Disable responsive for better performance
                maintainAspectRatio: false,
                animation: false, // Disable animations for large datasets
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // Store chart reference for cleanup
        this.charts.set('difficulty', chart);

        // Enable responsiveness after initial render
        setTimeout(() => {
            chart.options.responsive = true;
            chart.resize();
        }, 100);
    }

    createOptimizedQualityChart(ctx, qualityMetrics) {
        // Clean up existing chart
        const existingChart = this.charts.get('quality');
        if (existingChart) {
            existingChart.destroy();
        }

        const scores = qualityMetrics?.per_question_scores || [];
        
        // Limit data points to prevent performance issues
        const maxDataPoints = 50;
        const displayScores = scores.length > maxDataPoints 
            ? this.decimateData(scores, maxDataPoints)
            : scores;

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: displayScores.map((_, i) => `Q${i + 1}`),
                datasets: [{
                    label: 'Quality Score',
                    data: displayScores,
                    backgroundColor: displayScores.map(score =>
                        score >= 0.8 ? '#10B981' :
                            score >= 0.6 ? '#F59E0B' : '#EF4444'
                    )
                }]
            },
            options: {
                responsive: false, // Disable initially
                maintainAspectRatio: false,
                animation: false, // Disable for performance
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        // Store chart reference for cleanup
        this.charts.set('quality', chart);

        // Enable responsiveness after initial render
        setTimeout(() => {
            chart.options.responsive = true;
            chart.resize();
        }, 100);
    }

    decimateData(data, maxPoints) {
        if (data.length <= maxPoints) return data;
        
        const step = Math.ceil(data.length / maxPoints);
        const decimated = [];
        
        for (let i = 0; i < data.length; i += step) {
            decimated.push(data[i]);
        }
        
        return decimated;
    }

    // Legacy methods for backward compatibility
    createDifficultyChart(distribution) {
        const ctx = document.getElementById('difficultyChart')?.getContext('2d');
        if (ctx) {
            this.createOptimizedDifficultyChart(ctx, distribution);
        }
    }

    createQualityChart(qualityMetrics) {
        const ctx = document.getElementById('qualityChart')?.getContext('2d');
        if (ctx) {
            this.createOptimizedQualityChart(ctx, qualityMetrics);
        }
    }

    displayLowQualityQuestions(lowQualityQuestions) {
        const container = document.getElementById('lowQualityQuestions');

        if (lowQualityQuestions.length === 0) {
            container.innerHTML = '<p class="text-green-600">All questions meet quality standards! 🎉</p>';
            return;
        }

        container.innerHTML = lowQualityQuestions.map(q => `
            <div class="flex justify-between items-center p-3 bg-yellow-50 border border-yellow-200 rounded">
                <div>
                    <span class="font-medium">Question ${q.index + 1}</span>
                    <span class="text-sm text-gray-600 ml-2">Quality: ${(q.quality_score * 100).toFixed(0)}%</span>
                </div>
                <button onclick="assessmentGenerator.regenerateSpecificQuestion(${q.index})" 
                        class="px-3 py-1 bg-yellow-500 hover:bg-yellow-600 text-white rounded text-sm">
                    Regenerate
                </button>
            </div>
        `).join('');
    }

    async regenerateLowQuality() {
        if (!this.taskId) return;

        try {
            this.showInfo('Regenerating low quality questions...');
            
            const response = await fetch(`/api/regenerate/${this.taskId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    regenerate_all_below_threshold: true,
                    quality_threshold: 0.6
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.showSuccess(`Regenerated ${result.regenerated_count} questions`);
                
                // Refresh the assessment data by fetching the latest results
                await this.refreshAssessmentData();
                
                // Refresh the display and analytics
                if (this.lastAssessment) {
                    this.displayResults(this.lastAssessment);
                }
                this.loadAnalytics();
            } else {
                const error = await response.json();
                this.showError(`Regeneration failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Regeneration error:', error);
            this.showError('Failed to regenerate questions');
        }
    }

    async refreshAssessmentData() {
        if (!this.taskId) return;
        
        try {
            const response = await fetch(`/api/status/${this.taskId}`);
            if (response.ok) {
                const status = await response.json();
                if (status.status === 'ready' && status.result) {
                    this.lastAssessment = status.result;
                }
            }
        } catch (error) {
            console.error('Failed to refresh assessment data:', error);
        }
    }

    async regenerateSpecificQuestion(questionIndex) {
        if (!this.taskId) return;

        try {
            this.showInfo(`Regenerating question ${questionIndex + 1}...`);
            
            const response = await fetch(`/api/regenerate/${this.taskId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question_indices: [questionIndex]
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.showSuccess('Question regenerated successfully');
                
                // Refresh the assessment data by fetching the latest results
                await this.refreshAssessmentData();
                
                // Refresh the display and analytics
                if (this.lastAssessment) {
                    this.displayResults(this.lastAssessment);
                }
                this.loadAnalytics();
            } else {
                const error = await response.json();
                this.showError(`Question regeneration failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Question regeneration error:', error);
            this.showError('Failed to regenerate question');
        }
    }
}

// Initialize the application
let assessmentGenerator;

document.addEventListener('DOMContentLoaded', function () {
    assessmentGenerator = new AssessmentGenerator();
    console.log('🚀 STEM Assessment Generator initialized successfully!');
});

// Global functions for HTML onclick handlers
window.startOver = () => assessmentGenerator.startOver();
window.hideError = () => assessmentGenerator.hideError();
window.hideInfo = () => assessmentGenerator.hideInfo();
window.hideSuccess = () => assessmentGenerator.hideSuccess();
window.exportQuestions = () => assessmentGenerator.exportQuestions();
window.previewExport = () => assessmentGenerator.previewExport();
window.regenerateLowQuality = () => assessmentGenerator.regenerateLowQuality();
