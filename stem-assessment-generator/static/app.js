// STEM Assessment Generator Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorMessage = document.getElementById('errorMessage');
    const generateBtn = document.getElementById('generateBtn');
    const questionsContainer = document.getElementById('questionsContainer');

    // File upload drag and drop functionality
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateUploadArea(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            updateUploadArea(e.target.files[0]);
        }
    });
    
    function updateUploadArea(file) {
        const uploadText = uploadArea.querySelector('.upload-text');
        const uploadIcon = uploadArea.querySelector('.upload-icon');
        
        if (file && file.type === 'application/pdf') {
            uploadText.textContent = `Selected: ${file.name}`;
            uploadIcon.textContent = '✅';
            uploadArea.style.borderColor = '#4CAF50';
            uploadArea.style.backgroundColor = '#e8f5e8';
        } else {
            showError('Please select a valid PDF file');
            resetUploadArea();
        }
    }
    
    function resetUploadArea() {
        const uploadText = uploadArea.querySelector('.upload-text');
        const uploadIcon = uploadArea.querySelector('.upload-icon');
        
        uploadText.textContent = 'Drop your PDF here or click to browse';
        uploadIcon.textContent = '📄';
        uploadArea.style.borderColor = '#ddd';
        uploadArea.style.backgroundColor = 'transparent';
    }
    
    // Form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        
        const formData = new FormData();
        const file = fileInput.files[0];
        
        if (!file) {
            showError('Please select a PDF file');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            showError('File size must be less than 10MB');
            return;
        }
        
        formData.append('file', file);
        formData.append('num_questions', document.getElementById('numQuestions').value);
        formData.append('difficulty', document.getElementById('difficulty').value);
        
        try {
            showLoading();
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error occurred');
            }
            
            const result = await response.json();
            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An error occurred while processing your file');
            hideLoading();
        }
    });
    
    function showLoading() {
        document.querySelector('.form-section').style.display = 'none';
        resultsSection.style.display = 'none';
        loadingSection.style.display = 'block';
        generateBtn.disabled = true;
    }
    
    function hideLoading() {
        loadingSection.style.display = 'none';
        document.querySelector('.form-section').style.display = 'block';
        generateBtn.disabled = false;
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(hideError, 5000); // Auto-hide after 5 seconds
    }
    
    function hideError() {
        errorMessage.style.display = 'none';
    }
    
    function displayResults(data) {
        hideLoading();
        
        if (!data.questions || data.questions.length === 0) {
            showError('No questions were generated. Please try with a different document.');
            return;
        }
        
        questionsContainer.innerHTML = '';
        
        data.questions.forEach((question, index) => {
            const questionCard = createQuestionCard(question, index + 1);
            questionsContainer.appendChild(questionCard);
        });
        
        document.querySelector('.form-section').style.display = 'none';
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function createQuestionCard(question, questionNumber) {
        const card = document.createElement('div');
        card.className = 'question-card';
        
        // Create all options (correct answer + distractors)
        const allOptions = [
            { text: question.correct_answer, isCorrect: true },
            ...question.distractors.map(d => ({ text: d.text, isCorrect: false, explanation: d.explanation }))
        ];
        
        // Shuffle options
        shuffleArray(allOptions);
        
        const optionsHtml = allOptions.map((option, index) => {
            const letter = String.fromCharCode(65 + index); // A, B, C, D
            return `
                <li class="option" data-correct="${option.isCorrect}" onclick="selectOption(this)">
                    <strong>${letter}.</strong> ${option.text}
                </li>
            `;
        }).join('');
        
        card.innerHTML = `
            <div class="question-header">
                <span class="question-number">Question ${questionNumber}</span>
                <span class="difficulty-badge difficulty-${question.difficulty}">${question.difficulty.toUpperCase()}</span>
            </div>
            
            <div class="question-text">${question.question_text}</div>
            
            <ul class="options">
                ${optionsHtml}
            </ul>
            
            <div class="explanation" style="display: none;">
                <strong>Explanation:</strong> ${question.explanation}
            </div>
        `;
        
        return card;
    }
    
    // Global function for option selection
    window.selectOption = function(optionElement) {
        const card = optionElement.closest('.question-card');
        const allOptions = card.querySelectorAll('.option');
        const explanation = card.querySelector('.explanation');
        
        // Remove previous selections
        allOptions.forEach(opt => {
            opt.classList.remove('correct', 'incorrect');
        });
        
        // Mark selected option
        const isCorrect = optionElement.dataset.correct === 'true';
        optionElement.classList.add(isCorrect ? 'correct' : 'incorrect');
        
        // Show correct answer if wrong selection
        if (!isCorrect) {
            allOptions.forEach(opt => {
                if (opt.dataset.correct === 'true') {
                    opt.classList.add('correct');
                }
            });
        }
        
        // Show explanation
        explanation.style.display = 'block';
        
        // Disable further clicks
        allOptions.forEach(opt => {
            opt.style.pointerEvents = 'none';
        });
    };
    
    // Global function to start over
    window.startOver = function() {
        // Reset form
        uploadForm.reset();
        resetUploadArea();
        
        // Hide results and show form
        resultsSection.style.display = 'none';
        document.querySelector('.form-section').style.display = 'block';
        
        // Clear questions
        questionsContainer.innerHTML = '';
        
        // Hide error messages
        hideError();
        
        // Scroll to top
        document.querySelector('.header').scrollIntoView({ behavior: 'smooth' });
    };
    
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
    
    // Utility function to format text
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Auto-resize text areas (if needed)
    function autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (uploadForm.style.display !== 'none') {
                uploadForm.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to start over
        if (e.key === 'Escape') {
            if (resultsSection.style.display !== 'none') {
                startOver();
            }
        }
    });
    
    // Progress tracking (optional enhancement)
    function updateProgress(current, total) {
        // Implementation for progress tracking during generation
        // Could be added for better UX
    }
    
    // Accessibility enhancements
    function announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
    
    // Add focus management for better accessibility
    function manageFocus() {
        // Focus management for dynamic content
        const firstQuestion = document.querySelector('.question-card');
        if (firstQuestion) {
            firstQuestion.focus();
        }
    }
    
    console.log('STEM Assessment Generator initialized successfully!');
});
