"""
Question and distractor generation using OpenAI API
"""
import openai
import json
import uuid
from typing import List, Dict, Any
import random

from config import settings
from models import Question, Distractor, QuestionType, DifficultyLevel, AssessmentResponse
from patterns import get_misconception_patterns, get_distractor_templates

class AssessmentGenerator:
    """Generates assessment questions and distractors using AI"""
    
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.misconception_patterns = get_misconception_patterns()
    
    async def generate_assessment(
        self, 
        retriever: Any, 
        num_questions: int = 5,
        difficulty: str = "medium"
    ) -> AssessmentResponse:
        """
        Generate a complete assessment with questions and distractors
        
        Args:
            retriever: RAG retriever for content
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            AssessmentResponse with generated questions
        """
        questions = []
        
        # Get relevant content chunks for question generation
        content_chunks = await self._get_diverse_content(retriever, num_questions)
        
        for i, content in enumerate(content_chunks):
            try:
                question = await self._generate_single_question(
                    content, 
                    difficulty,
                    i + 1
                )
                if question:
                    questions.append(question)
            except Exception as e:
                print(f"Error generating question {i+1}: {str(e)}")
                continue
        
        metadata = {
            "generation_model": self.model,
            "difficulty": difficulty,
            "total_requested": num_questions,
            "successfully_generated": len(questions)
        }
        
        return AssessmentResponse(
            questions=questions,
            metadata=metadata,
            total_questions=len(questions)
        )
    
    async def _get_diverse_content(self, retriever: Any, num_questions: int) -> List[str]:
        """Get diverse content chunks for question generation"""
        # Generate diverse queries to get varied content
        queries = [
            "key concepts and definitions",
            "important principles and laws",
            "examples and applications",
            "problem solving methods",
            "common misconceptions"
        ]
        
        content_chunks = []
        for query in queries[:num_questions]:
            try:
                docs = retriever.get_relevant_documents(query)
                if docs:
                    content_chunks.append(docs[0].page_content)
            except:
                continue
        
        # If we don't have enough diverse content, get more with similarity search
        while len(content_chunks) < num_questions:
            try:
                docs = retriever.get_relevant_documents("important information")
                for doc in docs:
                    if doc.page_content not in content_chunks:
                        content_chunks.append(doc.page_content)
                        if len(content_chunks) >= num_questions:
                            break
                break
            except:
                break
        
        return content_chunks[:num_questions]
    
    async def _generate_single_question(
        self, 
        content: str, 
        difficulty: str,
        question_num: int
    ) -> Question:
        """Generate a single question with distractors"""
        
        prompt = self._create_question_prompt(content, difficulty)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator creating assessment questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the response
            question_data = self._parse_question_response(response.choices[0].message.content)
            
            # Generate distractors
            distractors = await self._generate_distractors(
                question_data["question_text"],
                question_data["correct_answer"],
                content
            )
            
            return Question(
                id=str(uuid.uuid4()),
                question_text=question_data["question_text"],
                question_type=QuestionType.MULTIPLE_CHOICE,
                difficulty=DifficultyLevel(difficulty),
                correct_answer=question_data["correct_answer"],
                distractors=distractors,
                explanation=question_data.get("explanation", ""),
                source_context=content[:200] + "...",
                bloom_taxonomy_level=question_data.get("bloom_level", "remember")
            )
            
        except Exception as e:
            print(f"Error in question generation: {str(e)}")
            return None
    
    def _create_question_prompt(self, content: str, difficulty: str) -> str:
        """Create prompt for question generation"""
        return f"""
Based on the following educational content, create ONE high-quality multiple choice question.

Content:
{content}

Requirements:
- Difficulty level: {difficulty}
- Question should test understanding, not just memorization
- Provide the correct answer
- Include a brief explanation
- Specify Bloom's taxonomy level (remember, understand, apply, analyze, evaluate, create)

Format your response as:
QUESTION: [question text]
CORRECT_ANSWER: [correct answer]
EXPLANATION: [brief explanation]
BLOOM_LEVEL: [taxonomy level]

Make sure the question is clear, unambiguous, and educational.
"""
    
    def _parse_question_response(self, response: str) -> Dict[str, str]:
        """Parse the AI response into structured data"""
        lines = response.strip().split('\n')
        result = {}
        
        for line in lines:
            if line.startswith('QUESTION:'):
                result['question_text'] = line.replace('QUESTION:', '').strip()
            elif line.startswith('CORRECT_ANSWER:'):
                result['correct_answer'] = line.replace('CORRECT_ANSWER:', '').strip()
            elif line.startswith('EXPLANATION:'):
                result['explanation'] = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('BLOOM_LEVEL:'):
                result['bloom_level'] = line.replace('BLOOM_LEVEL:', '').strip()
        
        return result
    
    async def _generate_distractors(
        self, 
        question: str, 
        correct_answer: str, 
        content: str
    ) -> List[Distractor]:
        """Generate plausible distractors based on common misconceptions"""
        
        distractor_prompt = f"""
Create 3 plausible but incorrect multiple choice distractors for this question.
Base the distractors on common student misconceptions and errors.

Question: {question}
Correct Answer: {correct_answer}
Context: {content[:300]}

For each distractor, provide:
1. The incorrect answer text
2. The type of misconception it represents
3. Why a student might choose this answer

Format:
DISTRACTOR_1: [text] | MISCONCEPTION: [type] | REASON: [explanation]
DISTRACTOR_2: [text] | MISCONCEPTION: [type] | REASON: [explanation]  
DISTRACTOR_3: [text] | MISCONCEPTION: [type] | REASON: [explanation]
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at understanding student misconceptions."},
                    {"role": "user", "content": distractor_prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            return self._parse_distractors(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error generating distractors: {str(e)}")
            return self._create_fallback_distractors()
    
    def _parse_distractors(self, response: str) -> List[Distractor]:
        """Parse AI response into Distractor objects"""
        distractors = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.startswith('DISTRACTOR_'):
                parts = line.split(' | ')
                if len(parts) >= 3:
                    text = parts[0].split(': ', 1)[1] if ': ' in parts[0] else parts[0]
                    misconception = parts[1].replace('MISCONCEPTION: ', '')
                    reason = parts[2].replace('REASON: ', '')
                    
                    distractors.append(Distractor(
                        text=text,
                        misconception_type=misconception,
                        explanation=reason
                    ))
        
        return distractors[:3]  # Return max 3 distractors
    
    def _create_fallback_distractors(self) -> List[Distractor]:
        """Create fallback distractors if AI generation fails"""
        return [
            Distractor(
                text="Option A",
                misconception_type="general_error",
                explanation="Common alternative answer"
            ),
            Distractor(
                text="Option B", 
                misconception_type="general_error",
                explanation="Plausible incorrect choice"
            ),
            Distractor(
                text="Option C",
                misconception_type="general_error", 
                explanation="Alternative interpretation"
            )
        ]
