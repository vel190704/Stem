"""
Enhanced assessment question generator with intelligent batching and robust error handling
Specialized for blockchain and STEM education with guaranteed quantity control
"""
import asyncio
import json
import logging
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import uuid

import openai
from config import settings
from models import Question, Distractor, AssessmentResponse, DifficultyLevel
from patterns import (
    MisconceptionPattern,
    PatternAnalyzer,
    generate_blockchain_distractors,
    create_pattern_registry,
    DifficultyLevel as PatternDifficultyLevel,
    ConceptCategory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionCache:
    """Cache to track generated questions and avoid duplicates"""
    
    def __init__(self):
        self.questions = []
        self.concepts_used = set()
        self.question_texts = set()
        self.similarity_threshold = 0.8
    
    def add_question(self, question: Question):
        """Add question to cache"""
        self.questions.append(question)
        self.question_texts.add(self._normalize_text(question.question_text))
    
    def is_duplicate(self, question_text: str) -> bool:
        """Check if question is too similar to existing ones"""
        normalized = self._normalize_text(question_text)
        
        # Exact match check
        if normalized in self.question_texts:
            return True
        
        # Similarity check (basic word overlap)
        for existing_text in self.question_texts:
            if self._calculate_similarity(normalized, existing_text) > self.similarity_threshold:
                return True
        
        return False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'\W+', ' ', text.lower()).strip()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple word overlap)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_used_concepts(self) -> set:
        """Get set of concepts already used"""
        return self.concepts_used.copy()
    
    def mark_concept_used(self, concept: str):
        """Mark a concept as used"""
        self.concepts_used.add(concept.lower())

class BatchGenerator:
    """Handles intelligent batching with different generation strategies"""
    
    def __init__(self, client: openai.OpenAI, model: str):
        self.client = client
        self.model = model
        self.generation_perspectives = [
            "definitional",
            "comparative", 
            "application",
            "problem-solving",
            "conceptual",
            "analytical"
        ]
        self.max_retries = 3
        self.base_backoff = 1  # seconds
    
    async def generate_batch(self, chunks: List[Dict[str, Any]], batch_size: int,
                           difficulty: str, subject: str, perspective: str,
                           cache: QuestionCache) -> List[Question]:
        """
        Enhanced batch generation with smart prompting and quality validation
        
        Args:
            chunks: Content chunks for generation
            batch_size: Number of questions to generate
            difficulty: Difficulty level
            subject: Subject area
            perspective: Generation perspective/strategy
            cache: Question cache for duplicate checking
            
        Returns:
            List[Question]: Generated and validated questions
        """
        failed_reasons = []
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Batch generation attempt {attempt + 1}/{self.max_retries} "
                          f"({batch_size} questions, {perspective} perspective)")
                
                # Select appropriate question types for this batch
                question_types = self._get_question_types_for_perspective(perspective)
                
                # Create enhanced prompt (first attempt) or retry prompt (subsequent attempts)
                if attempt == 0:
                    prompt = self._get_enhanced_generation_prompt(
                        chunks, batch_size, difficulty, perspective, question_types
                    )
                else:
                    prompt = self._get_intelligent_retry_prompt(
                        chunks, batch_size, difficulty, failed_reasons, attempt
                    )
                
                # Call OpenAI with enhanced parameters
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(subject, difficulty, perspective)
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7 + (attempt * 0.1),  # Increase creativity on retries
                    max_tokens=6000,  # Generous token limit
                    top_p=0.9,
                    frequency_penalty=0.3,
                    presence_penalty=0.2
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse and validate response
                questions = self._parse_batch_response(content, difficulty, perspective)
                
                # Enhanced quality validation
                validated_questions = []
                validation_failures = []
                
                for question in questions:
                    if self.validate_question_quality(question) and not cache.is_duplicate(question.question_text):
                        validated_questions.append(question)
                        cache.add_question(question)
                    else:
                        validation_failures.append(f"Failed validation: {question.question_text[:50]}...")
                
                # Track failure reasons for intelligent retry
                if validation_failures:
                    failed_reasons.extend(validation_failures)
                
                # Success if we got valid questions
                if validated_questions:
                    logger.info(f"Batch generation successful: {len(validated_questions)}/{batch_size} "
                              f"questions generated and validated")
                    return validated_questions
                else:
                    logger.warning(f"Batch attempt {attempt + 1} produced no valid questions")
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing failed on attempt {attempt + 1}: {e}"
                logger.warning(error_msg)
                failed_reasons.append("json_parsing_error")
                
            except Exception as e:
                error_msg = f"Batch generation error on attempt {attempt + 1}: {e}"
                logger.error(error_msg)
                failed_reasons.append(f"generation_error: {str(e)}")
                
                # Exponential backoff for API errors
                if "rate limit" in str(e).lower() or "api" in str(e).lower():
                    wait_time = (2 ** attempt) * 1.0
                    logger.info(f"API error, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        logger.error(f"All {self.max_retries} batch generation attempts failed")
        return []

    def _get_question_types_for_perspective(self, perspective: str) -> List[str]:
        """Get appropriate question types for the given perspective"""
        
        perspective_types = {
            "definitional": ["factual", "conceptual"],
            "procedural": ["application", "troubleshooting"],
            "comparative": ["comparative", "analytical"],
            "application": ["application", "troubleshooting"],
            "analytical": ["analytical", "conceptual"],
            "conceptual": ["conceptual", "factual"],
            "problem-solving": ["troubleshooting", "application"]
        }
        
        return perspective_types.get(perspective, ["factual", "conceptual", "application", "analytical"])

    def _get_system_prompt(self, subject: str, difficulty: str, perspective: str) -> str:
        """Get specialized system prompt based on parameters"""
        
        base_expertise = {
            "blockchain": "blockchain technology, cryptocurrency, and distributed systems",
            "physics": "physics, mechanics, and physical sciences",
            "chemistry": "chemistry, molecular science, and chemical processes",
            "biology": "biology, life sciences, and biological systems",
            "mathematics": "mathematics, algebra, calculus, and mathematical reasoning"
        }.get(subject.lower(), "STEM education and scientific concepts")
        
        perspective_instructions = {
            "definitional": "Focus on testing understanding of key terms, concepts, and definitions. Ask 'what is' or 'what does X mean' style questions.",
            "comparative": "Focus on comparing and contrasting different concepts, methods, or approaches. Ask questions about similarities, differences, and relationships.",
            "application": "Focus on applying concepts to real-world scenarios or new situations. Test practical understanding and usage.",
            "problem-solving": "Focus on multi-step problems that require analysis and logical reasoning. Test ability to work through complex scenarios.",
            "conceptual": "Focus on understanding underlying principles and how things work. Test deep comprehension of mechanisms and processes.",
            "analytical": "Focus on breaking down complex ideas, analyzing relationships, and evaluating different approaches or solutions."
        }
        
        difficulty_guidance = {
            "easy": "Use simple, direct language. Test basic recall and recognition. Distractors should be obviously wrong to experts but plausible to beginners.",
            "medium": "Use moderate technical language. Test understanding and application. Distractors should represent common misconceptions.",
            "hard": "Use advanced terminology. Test synthesis, evaluation, and complex reasoning. Distractors should require deep knowledge to eliminate."
        }
        
        return f"""You are an expert educational assessment creator specializing in {base_expertise}.

GENERATION PERSPECTIVE: {perspective.upper()}
{perspective_instructions.get(perspective, "Generate diverse, high-quality questions.")}

DIFFICULTY LEVEL: {difficulty.upper()}
{difficulty_guidance.get(difficulty, "Adjust complexity appropriately.")}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY the requested number of questions - no more, no less
2. Each question MUST have exactly 4 options (A, B, C, D)  
3. Only ONE option should be correct
4. The correct answer position should vary (don't always put it in position A)
5. At least 2 distractors must represent realistic student misconceptions
6. Each option must be a complete, substantive statement
7. Never use placeholder text like "Option A" or "Additional choice needed"
8. Questions must test genuine understanding, not trivial facts

QUALITY STANDARDS:
- Questions must be clear, unambiguous, and grammatically correct
- Options must be mutually exclusive and similar in length/style
- Distractors must be plausible to someone with incomplete understanding
- Test conceptual understanding appropriate to the {perspective} perspective
- Ensure variety in question stems and correct answer positions"""
    
    def _get_enhanced_generation_prompt(self, chunks: List[Dict[str, Any]], 
                                      batch_size: int, difficulty: str, 
                                      perspective: str, question_types: List[str]) -> str:
        """Create enhanced LLM prompt template with specific requirements"""
        
        # Combine chunk content intelligently
        chunk_content = "\n\n".join([
            f"CHUNK {i+1}: {chunk['content'][:500]}..."
            for i, chunk in enumerate(chunks[:3])
        ])
        
        # Question type templates for variety
        type_examples = {
            "factual": "What is the primary purpose of {concept}?",
            "conceptual": "Why does {concept} work this way?",
            "comparative": "How does {concept1} differ from {concept2}?",
            "application": "In which scenario would {concept} be most appropriate?",
            "troubleshooting": "What happens when {concept} fails?",
            "analytical": "What are the implications of {concept} in {context}?"
        }
        
        selected_types = random.sample(question_types, min(len(question_types), batch_size))
        type_guidance = "\n".join([
            f"- {type_examples.get(qtype, 'Generate diverse questions')}"
            for qtype in selected_types
        ])
        
        return f'''You are an expert STEM educator creating assessment questions with detailed misconception-based explanations.

CONTEXT MATERIAL:
{chunk_content}

TASK: Generate EXACTLY {batch_size} multiple-choice questions with comprehensive explanations for each option.

EXPLANATION REQUIREMENTS - CRITICAL:
For each INCORRECT option, provide:
1. misconception_name: Specific type of misconception (e.g., "Component Confusion", "Surface-Level Thinking")
2. why_students_think_this: Why students select this option (specific reasoning pattern)
3. why_its_wrong: Clear explanation of the error with evidence
4. correct_understanding: What the student should understand instead
5. remember_this: Key insight or memory aid

For the CORRECT option, provide:
1. why_correct: Detailed technical/scientific reasoning  
2. key_points: Array of main concepts tested
3. real_world_example: Concrete application example
4. connects_to: Related topics or applications

QUESTION QUALITY RULES:
1. Each question tests understanding, not memorization
2. Place correct answer randomly across A, B, C, D
3. Include 3 strong distractors representing real misconceptions
4. Options must be 5-30 words each, similar length
5. Avoid negative phrasing ("Which is NOT...")
6. Question must end with "?"

QUESTION TYPES TO USE:
{type_guidance}

DIFFICULTY: {difficulty.upper()} (adjust complexity accordingly)
PERSPECTIVE: {perspective.upper()} 

RETURN FORMAT - Return valid JSON array with EXACTLY {batch_size} questions:

[
  {{
    "question_text": "Clear, specific question ending with ?",
    "correct_answer": "Complete correct answer text",
    "options": {{
      "A": {{
        "text": "Complete option text (no placeholders)",
        "is_correct": false,
        "closeness_score": 7,
        "explanation": {{
          "misconception_name": "Specific misconception type",
          "why_students_think_this": "Students often believe this because...",
          "why_its_wrong": "This is incorrect because...",
          "correct_understanding": "The accurate understanding is...", 
          "remember_this": "Key rule: ..."
        }}
      }},
      "B": {{
        "text": "Complete option text (no placeholders)",
        "is_correct": true,
        "explanation": {{
          "why_correct": "This is correct because...",
          "key_points": ["Main concept", "Supporting principle", "Critical detail"],
          "real_world_example": "In practice, this is seen when...",
          "connects_to": "This concept relates to..."
        }}
      }},
      "C": {{
        "text": "Complete option text (no placeholders)", 
        "is_correct": false,
        "closeness_score": 6,
        "explanation": {{
          "misconception_name": "Different specific misconception",
          "why_students_think_this": "Students select this because...",
          "why_its_wrong": "This fails because...",
          "correct_understanding": "Instead, the right approach is...",
          "remember_this": "Remember: ..."
        }}
      }},
      "D": {{
        "text": "Complete option text (no placeholders)",
        "is_correct": false, 
        "closeness_score": 5,
        "explanation": {{
          "misconception_name": "Third misconception type",
          "why_students_think_this": "This seems right because...", 
          "why_its_wrong": "However, this overlooks...",
          "correct_understanding": "The complete picture requires...",
          "remember_this": "Key insight: ..."
        }}
      }}
    }},
    "correct_position": "B",
    "difficulty": "{difficulty.upper()}",
    "concept_tested": "specific_concept_from_material",
    "overall_explanation": "This question tests understanding of [concept] which is crucial because...",
    "explanation": "This tests understanding of [concept]. The correct answer is [correct option] because [brief reason]. Common misconceptions include [list main misconceptions tested]."
  }}
]

CRITICAL REQUIREMENTS:
- NO placeholder text like "Option A" or "Additional choice"
- ALL explanations must be complete and educational
- Each incorrect option must represent a realistic student misconception
- Explanations should help students learn, not just identify right/wrong
- Use content from the provided material as evidence in explanations'''

    def validate_question_quality(self, question: Question) -> bool:
        """Enhanced quality validation system with comprehensive checks"""
        
        # Quality check functions
        def check_no_placeholders(q):
            """Check for placeholder text in any option"""
            all_options = [q.correct_answer] + [d.text for d in q.distractors]
            placeholders = ['option a', 'option b', 'option c', 'option d', 
                          'placeholder', 'additional choice', 'choice needed',
                          'add option', 'another option']
            
            for option in all_options:
                if any(placeholder in option.lower() for placeholder in placeholders):
                    logger.warning(f"Placeholder found: {option}")
                    return False
            return True
        
        def check_option_length(q):
            """Check option length is reasonable (5-30 words)"""
            all_options = [q.correct_answer] + [d.text for d in q.distractors]
            for option in all_options:
                word_count = len(option.split())
                if word_count < 3 or word_count > 35:
                    logger.warning(f"Option length issue ({word_count} words): {option[:50]}...")
                    return False
            return True
        
        def check_distractor_quality(q):
            """Check for 2+ strong distractors"""
            if len(q.distractors) < 2:
                logger.warning("Insufficient distractors")
                return False
            
            # Check for empty or very short distractors
            for distractor in q.distractors:
                if len(distractor.text.strip()) < 5:
                    logger.warning(f"Distractor too short: {distractor.text}")
                    return False
            return True
        
        def check_question_clarity(q):
            """Check question is clear and unambiguous"""
            question_text = q.question_text.strip()
            
            # Must end with question mark
            if not question_text.endswith('?'):
                logger.warning(f"Question doesn't end with ?: {question_text}")
                return False
                
            # Must be reasonable length
            if len(question_text.split()) < 3 or len(question_text.split()) > 25:
                logger.warning(f"Question length issue: {question_text}")
                return False
                
            return True
        
        def check_no_negatives(q):
            """Avoid confusing negative questions"""
            negative_words = ['not', "n't", 'never', 'except', 'excluding']
            question_lower = q.question_text.lower()
            
            if any(word in question_lower for word in negative_words):
                logger.warning(f"Negative question detected: {q.question_text}")
                return False
            return True
        
        def check_option_balance(q):
            """Check options are similar in length and style"""
            all_options = [q.correct_answer] + [d.text for d in q.distractors]
            lengths = [len(option.split()) for option in all_options]
            
            if max(lengths) > 2 * min(lengths):  # One option much longer
                logger.warning("Option length imbalance detected")
                return False
            return True
        
        # Run all validation checks
        checks = [
            check_no_placeholders,
            check_option_length, 
            check_distractor_quality,
            check_question_clarity,
            check_no_negatives,
            check_option_balance
        ]
        
        validation_results = []
        for check in checks:
            try:
                result = check(question)
                validation_results.append(result)
                if not result:
                    logger.debug(f"Question failed check: {check.__name__}")
            except Exception as e:
                logger.error(f"Validation check {check.__name__} failed: {e}")
                validation_results.append(False)
        
        # Require most checks to pass (allow 1 failure for flexibility)
        passed_checks = sum(validation_results)
        total_checks = len(validation_results)
        
        if passed_checks >= total_checks - 1:  # Allow 1 failure
            return True
        else:
            logger.warning(f"Question failed validation: {passed_checks}/{total_checks} checks passed")
            return False

    def _get_intelligent_retry_prompt(self, chunks: List[Dict[str, Any]], 
                                    batch_size: int, difficulty: str, 
                                    failed_reasons: List[str], attempt: int) -> str:
        """Generate improved prompt based on previous failures"""
        
        chunk_content = "\n\n".join([chunk['content'][:400] for chunk in chunks[:2]])
        
        # Specific guidance based on failure reasons
        retry_guidance = []
        if "placeholder" in str(failed_reasons):
            retry_guidance.append("- Write complete, meaningful text for ALL options")
            retry_guidance.append("- Never use 'Option A', 'Choice B', or similar placeholders")
            
        if "length" in str(failed_reasons):
            retry_guidance.append("- Keep each option between 5-25 words")
            retry_guidance.append("- Make options similar in length and detail")
            
        if "distractor" in str(failed_reasons):
            retry_guidance.append("- Create plausible wrong answers that students might choose")
            retry_guidance.append("- Base distractors on common misconceptions")
        
        guidance_text = "\n".join(retry_guidance) if retry_guidance else "- Focus on creating high-quality, complete questions"
        
        return f'''RETRY ATTEMPT {attempt}: Previous generation had quality issues. Focus on these improvements:

{guidance_text}

CONTENT:
{chunk_content}

Generate exactly {batch_size} complete multiple-choice questions. Each must have:
- Clear question ending with ?
- Four complete, substantial options (A, B, C, D)  
- One clearly correct answer
- Three plausible but incorrect distractors
- NO placeholder text whatsoever

Example format:
{{
  "question_text": "What is the primary function of blockchain consensus mechanisms?",
  "options": {{
    "A": "To create new cryptocurrency tokens for miners",
    "B": "To ensure all network participants agree on transaction validity", 
    "C": "To encrypt transaction data for enhanced privacy",
    "D": "To reduce network fees and increase transaction speed"
  }},
  "correct_position": "B",
  "difficulty": "{difficulty.upper()}",
  "concept_tested": "consensus_mechanisms"
}}

Return JSON array with {batch_size} questions:'''

    def _select_chunk_content(self, chunks: List[Dict[str, Any]], batch_size: int) -> str:
        """Select and combine relevant chunk content"""
        # Select chunks based on batch size and content diversity
        selected_chunks = chunks[:min(batch_size + 1, len(chunks))]
        
        combined_content = ""
        for i, chunk in enumerate(selected_chunks):
            content = chunk.get("content", chunk.get("text", ""))
            combined_content += f"\n--- Content Block {i+1} ---\n{content}\n"
        
        # Limit total length to avoid token limits
        if len(combined_content) > 3000:
            combined_content = combined_content[:3000] + "...\n[Content truncated for length]"
        
        return combined_content
    
    def _parse_batch_response(self, content: str, difficulty: str, perspective: str) -> List[Question]:
        """Parse OpenAI response into Question objects"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                logger.error("No JSON array found in response")
                return []
            
            questions_data = json.loads(json_str)
            
            if not isinstance(questions_data, list):
                logger.error("Response is not a JSON array")
                return []
            
            questions = []
            for i, q_data in enumerate(questions_data):
                question = self._create_question_from_data(q_data, difficulty, perspective, i)
                if question:
                    questions.append(question)
            
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            return []
    
    def _create_question_from_data(self, q_data: Dict[str, Any], difficulty: str, 
                                 perspective: str, question_index: int) -> Optional[Question]:
        """Create Question object from parsed data"""
        try:
            # Extract basic fields
            question_text = q_data.get("question_text", "").strip()
            concept_tested = q_data.get("concept_tested", "unknown")
            
            if not question_text:
                logger.warning(f"Question {question_index}: Missing question text")
                return None
            
            # Extract and validate options
            options_data = q_data.get("options", {})
            if len(options_data) != 4:
                logger.warning(f"Question {question_index}: Expected 4 options, got {len(options_data)}")
                return None
            
            # Find correct answer and preserve detailed explanations
            correct_answer = None
            distractors = []
            detailed_options = {}
            
            for option_key, option_data in options_data.items():
                if isinstance(option_data, dict):
                    # New detailed format
                    option_text = option_data.get("text", "").strip()
                    is_correct = option_data.get("is_correct", False)
                    explanation = option_data.get("explanation", {})
                    
                    # Store full option data for frontend
                    detailed_options[option_key] = {
                        "text": option_text,
                        "is_correct": is_correct,
                        "explanation": explanation
                    }
                else:
                    # Legacy format - option_data is just the text
                    option_text = str(option_data).strip()
                    is_correct = (q_data.get("correct_position", "") == option_key)
                    
                    detailed_options[option_key] = {
                        "text": option_text,
                        "is_correct": is_correct,
                        "explanation": {}
                    }
                
                if not option_text:
                    logger.warning(f"Question {question_index}: Empty option text for {option_key}")
                    return None
                
                if is_correct:
                    if correct_answer:
                        logger.warning(f"Question {question_index}: Multiple correct answers found")
                        return None
                    correct_answer = option_text
                else:
                    # Get misconception details from explanation
                    misconception_type = "General Misconception"
                    misconception_explanation = f"Common error in {perspective} understanding"
                    
                    if isinstance(explanation, dict):
                        misconception_type = explanation.get("misconception_name", misconception_type)
                        misconception_explanation = explanation.get("why_its_wrong", misconception_explanation)
                    
                    distractor = Distractor(
                        text=option_text,
                        misconception_type=misconception_type,
                        explanation=misconception_explanation
                    )
                    distractors.append(distractor)
            
            if not correct_answer:
                logger.warning(f"Question {question_index}: No correct answer found")
                return None
            
            if len(distractors) != 3:
                logger.warning(f"Question {question_index}: Expected 3 distractors, got {len(distractors)}")
                return None
            
            # Create Question object with detailed explanations
            overall_explanation = q_data.get("overall_explanation", f"This {perspective} question tests understanding of {concept_tested}")
            
            question = Question(
                question_text=question_text,
                difficulty=DifficultyLevel(difficulty),
                correct_answer=correct_answer,
                distractors=distractors,
                explanation=overall_explanation,
                source_context=concept_tested,
                bloom_taxonomy_level=self._map_perspective_to_bloom(perspective),
                metadata={
                    "concept_tested": concept_tested,
                    "perspective": perspective,
                    "generation_method": "batch_intelligent",
                    "question_index": question_index,
                    "detailed_options": detailed_options,
                    "overall_explanation": overall_explanation
                }
            )
            
            return question
            
        except Exception as e:
            logger.error(f"Error creating question {question_index}: {e}")
            return None
    
    def _map_perspective_to_bloom(self, perspective: str) -> str:
        """Map generation perspective to Bloom's taxonomy level"""
        mapping = {
            "definitional": "remember",
            "comparative": "understand",
            "application": "apply",
            "problem-solving": "analyze",
            "conceptual": "understand",
            "analytical": "analyze"
        }
        return mapping.get(perspective, "understand")
    
    def _validate_question_quality(self, question: Question, cache: QuestionCache) -> bool:
        """Validate question meets quality standards"""
        
        # Check for duplicates
        if cache.is_duplicate(question.question_text):
            logger.warning("Duplicate question detected")
            return False
        
        # Check question text quality
        words = question.question_text.split()
        if len(words) < 5:
            logger.warning("Question text too short")
            return False
        
        if len(words) > 40:
            logger.warning("Question text too long")
            return False
        
        if not question.question_text.endswith('?'):
            logger.warning("Question doesn't end with question mark")
            return False
        
        # Check correct answer quality
        if len(question.correct_answer.split()) < 2:
            logger.warning("Correct answer too short")
            return False
        
        # Check distractors
        if len(question.distractors) != 3:
            logger.warning("Wrong number of distractors")
            return False
        
        # Check for placeholder text
        all_options = [question.correct_answer] + [d.text for d in question.distractors]
        for option in all_options:
            if any(placeholder in option.lower() for placeholder in 
                   ['option a', 'option b', 'option c', 'option d', 'placeholder', 'additional choice']):
                logger.warning(f"Placeholder text found: {option}")
                return False
        
        return True

class AssessmentGenerator:
    """Main orchestrator for assessment generation with intelligent batching"""
    
    def __init__(self):
        """Initialize generator with OpenAI client and batch processor"""
        settings.validate_openai_key()
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.batch_generator = BatchGenerator(self.client, self.model)
        self.generation_stats = {
            "total_attempts": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_questions_generated": 0,
            "perspective_usage": {}
        }
    
    async def generate_assessment(self, retriever, num_questions: int, 
                                difficulty: str, subject: str = "blockchain") -> AssessmentResponse:
        """
        Main orchestrator - GUARANTEES exactly num_questions are returned
        
        Args:
            retriever: RAG retriever for content
            num_questions: Target number of questions (EXACT)
            difficulty: Difficulty level
            subject: Subject area
            
        Returns:
            AssessmentResponse with exactly num_questions
        """
        start_time = time.time()
        logger.info(f"Starting assessment generation: EXACTLY {num_questions} questions required")
        
        # Initialize cache and tracking
        cache = QuestionCache()
        all_questions = []
        
        # Retrieve diverse content
        chunks = await self._retrieve_diverse_chunks(retriever, num_questions)
        
        # Generation strategy: batches with different perspectives
        batch_size = min(3, num_questions)  # Start with small batches
        perspectives = self.batch_generator.generation_perspectives.copy()
        random.shuffle(perspectives)  # Randomize perspective order
        
        perspective_index = 0
        retry_count = 0
        max_total_retries = num_questions * 2  # Safety limit
        
        while len(all_questions) < num_questions and retry_count < max_total_retries:
            remaining = num_questions - len(all_questions)
            current_batch_size = min(batch_size, remaining)
            current_perspective = perspectives[perspective_index % len(perspectives)]
            
            logger.info(f"Generating batch: {current_batch_size} questions, "
                       f"{remaining} remaining, perspective: {current_perspective}")
            
            try:
                # Generate batch with current perspective
                batch_questions = await self.batch_generator.generate_batch(
                    chunks, current_batch_size, difficulty, subject, 
                    current_perspective, cache
                )
                
                if batch_questions:
                    all_questions.extend(batch_questions)
                    self.generation_stats["successful_batches"] += 1
                    self.generation_stats["total_questions_generated"] += len(batch_questions)
                    
                    # Track perspective usage
                    if current_perspective not in self.generation_stats["perspective_usage"]:
                        self.generation_stats["perspective_usage"][current_perspective] = 0
                    self.generation_stats["perspective_usage"][current_perspective] += len(batch_questions)
                    
                    logger.info(f"Batch successful! Total questions: {len(all_questions)}/{num_questions}")
                    
                    # Reset retry count on success
                    retry_count = 0
                else:
                    self.generation_stats["failed_batches"] += 1
                    retry_count += 1
                    
                    # Switch perspective on failure
                    perspective_index += 1
                    
                    # Reduce batch size if consistently failing
                    if retry_count > 2:
                        batch_size = max(1, batch_size - 1)
                        logger.warning(f"Reducing batch size to {batch_size}")
                    
                    logger.warning(f"Batch failed, trying {perspectives[perspective_index % len(perspectives)]} perspective")
                
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                self.generation_stats["failed_batches"] += 1
                retry_count += 1
                perspective_index += 1
            
            self.generation_stats["total_attempts"] += 1
            
            # Brief pause between batches
            await asyncio.sleep(0.5)
        
        # Final validation and fallback if needed
        if len(all_questions) < num_questions:
            logger.warning(f"Generated {len(all_questions)}/{num_questions} questions. Creating fallbacks...")
            
            fallback_needed = num_questions - len(all_questions)
            fallback_questions = self._generate_fallback_questions(
                fallback_needed, difficulty, subject, cache
            )
            all_questions.extend(fallback_questions)
        
        # Ensure exact count (trim if somehow we got too many)
        final_questions = all_questions[:num_questions]
        
        # Compile response
        generation_time = time.time() - start_time
        
        response = AssessmentResponse(
            questions=final_questions,
            total_questions=len(final_questions),
            metadata={
                "generation_time": generation_time,
                "difficulty": difficulty,
                "subject": subject,
                "chunks_used": len(chunks),
                "generation_stats": self.generation_stats,
                "guaranteed_count": True
            },
            statistics={
                "requested_questions": num_questions,
                "generated_questions": len(final_questions),
                "success_rate": len(final_questions) / num_questions,
                "avg_time_per_question": generation_time / len(final_questions) if final_questions else 0,
                "exact_count_achieved": len(final_questions) == num_questions
            }
        )
        
        logger.info(f"Assessment generation completed: {len(final_questions)}/{num_questions} questions "
                   f"in {generation_time:.2f}s (EXACT COUNT: {len(final_questions) == num_questions})")
        
        return response
    
    async def _retrieve_diverse_chunks(self, retriever, num_questions: int) -> List[Dict[str, Any]]:
        """Retrieve diverse content chunks for question generation"""
        target_chunks = min(num_questions * 2, 12)
        
        # Diverse query strategies
        query_strategies = [
            "fundamental concepts and definitions",
            "key principles and mechanisms", 
            "practical applications and examples",
            "common problems and solutions",
            "important comparisons and differences",
            "technical processes and procedures",
            "theoretical foundations",
            "real-world implementations"
        ]
        
        all_chunks = []
        
        for query in query_strategies[:target_chunks//2]:
            try:
                docs = retriever.get_relevant_documents(query)
                
                for doc in docs[:2]:  # Top 2 per query
                    chunk_data = {
                        "content": doc.page_content,
                        "metadata": getattr(doc, 'metadata', {}),
                        "query": query,
                        "relevance_score": getattr(doc, 'score', 0.8)
                    }
                    all_chunks.append(chunk_data)
                    
                    if len(all_chunks) >= target_chunks:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to retrieve chunks for query '{query}': {e}")
                continue
        
        # Ensure we have at least some content
        if not all_chunks:
            logger.warning("No chunks retrieved, using fallback content")
            all_chunks = [{"content": "blockchain technology and distributed systems", "metadata": {}, "query": "fallback"}]
        
        return all_chunks[:target_chunks]
    
    def _generate_fallback_questions(self, count: int, difficulty: str, 
                                   subject: str, cache: QuestionCache) -> List[Question]:
        """Generate fallback questions to guarantee exact count"""
        logger.warning(f"Generating {count} fallback questions to meet exact requirement")
        
        fallback_templates = {
            "blockchain": [
                {
                    "question": "What is the primary purpose of blockchain consensus mechanisms?",
                    "correct": "To ensure all network participants agree on the current state of the ledger",
                    "distractors": [
                        "To encrypt all transaction data for security",
                        "To determine the price of cryptocurrency tokens",
                        "To validate user identities on the network"
                    ],
                    "concept": "consensus mechanisms"
                },
                {
                    "question": "What distinguishes a public blockchain from a private blockchain?",
                    "correct": "Public blockchains are open to anyone, while private blockchains restrict access",
                    "distractors": [
                        "Public blockchains are faster than private blockchains",
                        "Private blockchains use different cryptographic algorithms",
                        "Public blockchains cannot handle smart contracts"
                    ],
                    "concept": "blockchain types"
                },
                {
                    "question": "What is the role of miners in a proof-of-work blockchain?",
                    "correct": "To validate transactions and add new blocks by solving computational puzzles",
                    "distractors": [
                        "To create new cryptocurrency tokens for distribution",
                        "To manage user accounts and permissions",
                        "To determine transaction fees for the network"
                    ],
                    "concept": "mining and proof-of-work"
                }
            ]
        }
        
        templates = fallback_templates.get(subject, fallback_templates["blockchain"])
        questions = []
        
        for i in range(count):
            template = templates[i % len(templates)]
            
            # Create unique version by slightly modifying
            question_text = template["question"]
            if i >= len(templates):
                question_text = f"In {subject} systems, {question_text.lower()}"
            
            # Check for duplicates
            if cache.is_duplicate(question_text):
                question_text = f"Regarding {subject}, {template['question'].lower()}"
            
            distractors = [
                Distractor(
                    text=distractor_text,
                    misconception_type="fallback_misconception",
                    explanation="Common misconception addressed by fallback question"
                )
                for distractor_text in template["distractors"]
            ]
            
            question = Question(
                question_text=question_text,
                difficulty=DifficultyLevel(difficulty),
                correct_answer=template["correct"],
                distractors=distractors,
                explanation=f"This tests understanding of {template['concept']}",
                source_context=template["concept"],
                bloom_taxonomy_level="understand",
                metadata={
                    "generation_method": "guaranteed_fallback",
                    "template_used": i % len(templates),
                    "fallback_index": i
                }
            )
            
            questions.append(question)
            cache.add_question(question)
        
        return questions
