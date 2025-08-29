# Manual Testing Checklist for STEM Assessment Generator

## Pre-Testing Setup ✅

- [ ] **Environment Setup**
  - [ ] Virtual environment activated
  - [ ] All dependencies installed (`pip install -r requirements.txt`)
  - [ ] OpenAI API key configured (optional for basic testing)
  - [ ] Data directories created (`data/uploads/`, `data/vectordb/`)

- [ ] **Test Data Preparation**
  - [ ] Run `python sample_content_generator.py` to create test files
  - [ ] Verify test files exist in `test_data/` directory
  - [ ] Optional: Convert text files to actual PDFs for full testing

## Component Testing ✅

### PDF Processor Testing
- [ ] **Basic Functionality**
  - [ ] Import works: `python -c "from pdf_processor import PDFProcessor; print('OK')"`
  - [ ] Can extract text from sample files
  - [ ] Creates chunks with proper metadata
  - [ ] Handles different file formats gracefully

- [ ] **Quality Checks**
  - [ ] Chunks are reasonable length (not too short/long)
  - [ ] Metadata includes page numbers and chunk IDs
  - [ ] Text extraction preserves important formatting
  - [ ] No empty or corrupted chunks

### RAG Pipeline Testing
- [ ] **Basic Functionality**
  - [ ] Import works: `python -c "from rag_pipeline import RAGPipeline; print('OK')"`
  - [ ] Can process chunks and create embeddings
  - [ ] ChromaDB collection created successfully
  - [ ] Retrieval returns relevant results

- [ ] **Quality Checks**
  - [ ] Embedding generation completes without errors
  - [ ] Vector database persists between sessions
  - [ ] Retrieval results are relevant to queries
  - [ ] MMR diversity improves result variety

### Generator Testing
- [ ] **Basic Functionality**
  - [ ] Import works: `python -c "from generator import AssessmentGenerator; print('OK')"`
  - [ ] Can generate questions from retriever
  - [ ] Returns valid question format
  - [ ] Handles different difficulty levels

- [ ] **Quality Checks**
  - [ ] No placeholder text in questions or options
  - [ ] All options are complete sentences
  - [ ] Correct answer position varies (A, B, C, D distribution)
  - [ ] Questions test understanding, not just memorization

## Integration Testing ✅

### Full Pipeline Test
- [ ] **End-to-End Workflow**
  - [ ] Run `python test_full_pipeline.py`
  - [ ] PDF processing completes successfully
  - [ ] RAG pipeline setup works
  - [ ] Question generation produces valid output
  - [ ] No errors in the complete workflow

- [ ] **Performance Benchmarks**
  - [ ] PDF processing: < 5 seconds for 3-page document
  - [ ] RAG setup: < 10 seconds for 20 chunks
  - [ ] Question generation: < 3 seconds per question
  - [ ] Total pipeline: < 30 seconds for 10 questions

### Web Interface Testing
- [ ] **Application Startup**
  - [ ] Run `python app.py` without errors
  - [ ] Web interface loads at `http://localhost:8000`
  - [ ] Static files (CSS, JS) load correctly
  - [ ] No console errors in browser

- [ ] **File Upload Workflow**
  - [ ] Can select and upload PDF files
  - [ ] Progress indicator shows during processing
  - [ ] Status updates appear in real-time
  - [ ] Error messages are clear and helpful

- [ ] **Question Generation Workflow**
  - [ ] Can set number of questions (1-20)
  - [ ] Can select difficulty level
  - [ ] Generation process shows progress
  - [ ] Results display properly formatted questions

## Quality Validation ✅

### Question Quality Metrics
- [ ] **Content Quality**
  - [ ] All questions are complete and readable
  - [ ] No "Option A", "Option B" placeholder text
  - [ ] No "[INSERT_TEXT]" or similar placeholders
  - [ ] Options are similar length (not obviously unbalanced)

- [ ] **Educational Quality**
  - [ ] Questions test conceptual understanding
  - [ ] Distractors represent realistic misconceptions
  - [ ] Difficulty matches selected level
  - [ ] Questions cover variety of topics from source

- [ ] **Technical Quality**
  - [ ] Valid JSON structure in API responses
  - [ ] Correct answer marked properly
  - [ ] All required fields present
  - [ ] No duplicate questions in same assessment

### Distractor Analysis
- [ ] **Distractor Strength**
  - [ ] At least 2 strong distractors per question
  - [ ] Distractors are plausible but incorrect
  - [ ] Misconception types identified correctly
  - [ ] Closeness scores reasonable (6-9 range)

- [ ] **Pattern Application**
  - [ ] Run `python -c "from patterns import test_patterns; test_patterns()"`
  - [ ] Patterns apply to appropriate concepts
  - [ ] Generated distractors match pattern types
  - [ ] Variety in misconception categories used

## Error Handling Testing ✅

### Invalid Input Handling
- [ ] **File Upload Errors**
  - [ ] Handles non-PDF files gracefully
  - [ ] Shows error for files too large
  - [ ] Handles corrupted files without crashing
  - [ ] Clear error messages for unsupported formats

- [ ] **API Error Handling**
  - [ ] OpenAI API errors handled gracefully
  - [ ] Network timeout errors handled
  - [ ] Invalid request parameters rejected
  - [ ] Appropriate HTTP status codes returned

### Recovery Testing
- [ ] **Partial Failure Recovery**
  - [ ] Can retry failed question generation
  - [ ] Partial results saved during failures
  - [ ] Can resume interrupted processing
  - [ ] Task status accurately reflects state

## Performance Testing ✅

### Load Testing
- [ ] **Concurrent Operations**
  - [ ] Multiple file uploads don't interfere
  - [ ] Parallel question generation works
  - [ ] Database handles concurrent access
  - [ ] Memory usage stays reasonable

- [ ] **Scalability Testing**
  - [ ] Large PDFs (10+ pages) process correctly
  - [ ] High question counts (20+) generate successfully
  - [ ] Vector database grows without issues
  - [ ] Response times remain acceptable

## Manual Review Checklist ✅

### Generated Question Review
For each test run, manually review 5-10 questions:

- [ ] **Question Text**
  - [ ] Clear and unambiguous
  - [ ] Tests important concept
  - [ ] Appropriate for target audience
  - [ ] No grammatical errors

- [ ] **Options**
  - [ ] All options are complete
  - [ ] One clearly correct answer
  - [ ] Distractors are reasonable but wrong
  - [ ] Similar complexity across options

- [ ] **Educational Value**
  - [ ] Assesses understanding vs memorization
  - [ ] Covers key learning objectives
  - [ ] Difficulty appropriate for level
  - [ ] Useful for learning assessment

### System Behavior Review
- [ ] **User Experience**
  - [ ] Interface is intuitive
  - [ ] Progress feedback is helpful
  - [ ] Error messages are actionable
  - [ ] Results are easy to understand

- [ ] **System Reliability**
  - [ ] No crashes during normal use
  - [ ] Consistent behavior across test runs
  - [ ] Data persists correctly
  - [ ] Cleanup works properly

## Final Validation ✅

### Acceptance Criteria
- [ ] **Functional Requirements**
  - [ ] ✅ Processes PDF documents correctly
  - [ ] ✅ Generates specified number of questions
  - [ ] ✅ Questions have good quality distractors
  - [ ] ✅ Web interface is user-friendly
  - [ ] ✅ System handles errors gracefully

- [ ] **Quality Requirements**
  - [ ] ✅ <5% placeholder occurrence rate
  - [ ] ✅ >90% question generation success rate
  - [ ] ✅ Average 2.5+ distractors per question
  - [ ] ✅ Balanced correct answer distribution
  - [ ] ✅ Questions cover multiple concepts

- [ ] **Performance Requirements**
  - [ ] ✅ <2 seconds average per question
  - [ ] ✅ <30 seconds total for 10 questions
  - [ ] ✅ Handles 5MB PDF files
  - [ ] ✅ Reasonable memory usage (<2GB)

## Test Execution Commands

```bash
# Basic import tests
python -c "import app; print('App imports OK')"
python -c "import config; print('Config OK, OpenAI:', config.settings.is_openai_configured())"

# Generate sample data
python sample_content_generator.py

# Run component tests
python test_full_pipeline.py

# Run comprehensive test suite
python run_comprehensive_tests.py

# Quality validation (after generating assessment)
python quality_validator.py assessment_output.json

# Start web application
python app.py
```

## Common Issues and Solutions

### Setup Issues
- **Import errors**: Check virtual environment and dependencies
- **Directory errors**: Run app once to create required directories
- **OpenAI errors**: Verify API key or skip AI-dependent tests

### Quality Issues
- **Placeholder text**: Check prompt templates and validation
- **Poor distractors**: Review pattern application and misconception types
- **Unbalanced options**: Improve option length validation

### Performance Issues
- **Slow generation**: Check OpenAI API rate limits
- **Memory usage**: Monitor ChromaDB collection sizes
- **Timeout errors**: Increase timeout values in config

## Sign-off

- [ ] **Developer Testing Complete** (Date: _____, Signature: _____)
- [ ] **Integration Testing Complete** (Date: _____, Signature: _____)
- [ ] **Quality Validation Complete** (Date: _____, Signature: _____)
- [ ] **System Ready for Deployment** (Date: _____, Signature: _____)

---
*This checklist ensures comprehensive validation of the STEM Assessment Generator before deployment or demonstration.*
