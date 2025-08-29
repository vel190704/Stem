"""
Analytics module for tracking and improving question quality
Provides comprehensive quality metrics, assessment analysis, and performance tracking
"""
import sqlite3
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import math
import statistics

try:
    import textstat
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    print("⚠ Analytics dependencies not installed. Run: pip install textstat scikit-learn")

from models import Question, AssessmentResponse, MisconceptionPattern
from config import settings

class QuestionQualityAnalyzer:
    """Analyzes individual question quality across multiple dimensions"""
    
    def __init__(self):
        self.blooms_keywords = {
            "remember": ["define", "list", "name", "identify", "recall", "state"],
            "understand": ["explain", "describe", "summarize", "interpret", "classify"],
            "apply": ["use", "implement", "execute", "solve", "demonstrate"],
            "analyze": ["compare", "contrast", "distinguish", "examine", "categorize"],
            "evaluate": ["judge", "critique", "assess", "justify", "defend"],
            "create": ["design", "construct", "develop", "formulate", "combine"]
        }
        
        self.clarity_indicators = {
            "positive": ["clear", "specific", "precise", "exact", "definite"],
            "negative": ["unclear", "ambiguous", "vague", "confusing", "multiple"]
        }

    def analyze_question_quality(self, question: Question) -> Dict[str, Any]:
        """
        Comprehensive quality analysis of a single question
        """
        if not ANALYTICS_AVAILABLE:
            return self._basic_analysis(question)
        
        analysis = {
            "readability": self._calculate_readability(question.question_text),
            "distractor_quality": self._analyze_distractors(question),
            "cognitive_level": self._classify_blooms_taxonomy(question.question_text),
            "clarity_score": self._calculate_clarity_score(question),
            "estimated_difficulty": self._predict_difficulty(question),
            "language_quality": self._analyze_language_quality(question),
            "concept_focus": self._analyze_concept_focus(question),
            "overall_score": 0.0
        }
        
        # Calculate overall quality score (weighted average)
        weights = {
            "readability": 0.15,
            "distractor_quality": 0.30,
            "cognitive_level": 0.20,
            "clarity_score": 0.25,
            "language_quality": 0.10
        }
        
        analysis["overall_score"] = sum(
            analysis[metric].get("score", 0) * weight 
            for metric, weight in weights.items()
        )
        
        analysis["quality_grade"] = self._grade_quality(analysis["overall_score"])
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis

    def _basic_analysis(self, question: Question) -> Dict[str, Any]:
        """Basic analysis when full analytics not available"""
        return {
            "readability": {"score": 0.7, "level": "appropriate"},
            "distractor_quality": {
                "strong_count": len([d for d in question.distractors if d.closeness_score >= 6]),
                "average_closeness": statistics.mean([d.closeness_score for d in question.distractors]) if question.distractors else 0,
                "score": 0.7
            },
            "cognitive_level": {"level": "understand", "score": 0.7},
            "clarity_score": {"score": 0.8, "issues": []},
            "overall_score": 0.7,
            "quality_grade": "B",
            "recommendations": ["Install analytics dependencies for detailed analysis"]
        }

    def _calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calculate readability using multiple metrics"""
        flesch_score = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
        
        # Normalize to 0-1 scale where 0.7-0.9 is ideal for college level
        normalized_score = max(0, min(1, (flesch_score - 30) / 70))
        
        return {
            "flesch_score": flesch_score,
            "grade_level": flesch_grade,
            "score": normalized_score,
            "level": self._classify_readability_level(flesch_score),
            "appropriate": 8 <= flesch_grade <= 12  # College appropriate
        }

    def _classify_readability_level(self, flesch_score: float) -> str:
        """Classify readability level based on Flesch score"""
        if flesch_score >= 90:
            return "very_easy"
        elif flesch_score >= 80:
            return "easy"
        elif flesch_score >= 70:
            return "fairly_easy"
        elif flesch_score >= 60:
            return "standard"
        elif flesch_score >= 50:
            return "fairly_difficult"
        elif flesch_score >= 30:
            return "difficult"
        else:
            return "very_difficult"

    def _analyze_distractors(self, question: Question) -> Dict[str, Any]:
        """Analyze quality and distribution of distractors"""
        if not question.distractors:
            return {"strong_count": 0, "score": 0, "issues": ["No distractors found"]}
        
        closeness_scores = [d.closeness_score for d in question.distractors]
        misconception_types = [d.misconception_type for d in question.distractors]
        
        strong_distractors = [score for score in closeness_scores if score >= 6]
        weak_distractors = [score for score in closeness_scores if score < 4]
        
        diversity_score = len(set(misconception_types)) / len(misconception_types) if misconception_types else 0
        
        # Calculate overall distractor quality score
        avg_closeness = statistics.mean(closeness_scores)
        strong_ratio = len(strong_distractors) / len(closeness_scores)
        
        score = (avg_closeness / 10) * 0.6 + strong_ratio * 0.3 + diversity_score * 0.1
        
        return {
            "strong_count": len(strong_distractors),
            "weak_count": len(weak_distractors),
            "average_closeness": avg_closeness,
            "distribution": closeness_scores,
            "diversity": diversity_score,
            "misconception_types": misconception_types,
            "score": min(1.0, score),
            "issues": self._identify_distractor_issues(question)
        }

    def _identify_distractor_issues(self, question: Question) -> List[str]:
        """Identify specific issues with distractors"""
        issues = []
        
        if len(question.distractors) < 3:
            issues.append("Insufficient number of distractors")
        
        closeness_scores = [d.closeness_score for d in question.distractors]
        if sum(1 for score in closeness_scores if score >= 6) < 2:
            issues.append("Not enough strong distractors (closeness >= 6)")
        
        if statistics.stdev(closeness_scores) < 1.0 if len(closeness_scores) > 1 else False:
            issues.append("Distractors have similar closeness scores")
        
        misconception_types = [d.misconception_type for d in question.distractors]
        if len(set(misconception_types)) == 1:
            issues.append("All distractors use same misconception type")
        
        return issues

    def _classify_blooms_taxonomy(self, question_text: str) -> Dict[str, Any]:
        """Classify question according to Bloom's taxonomy"""
        text_lower = question_text.lower()
        
        level_scores = {}
        for level, keywords in self.blooms_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            level_scores[level] = score
        
        if not any(level_scores.values()):
            # Default classification based on question structure
            if any(word in text_lower for word in ["what is", "define", "identify"]):
                primary_level = "remember"
            elif any(word in text_lower for word in ["how", "why", "explain"]):
                primary_level = "understand"
            elif any(word in text_lower for word in ["compare", "difference"]):
                primary_level = "analyze"
            else:
                primary_level = "understand"
        else:
            primary_level = max(level_scores.items(), key=lambda x: x[1])[0]
        
        # Convert to numerical score (higher levels = higher scores)
        level_hierarchy = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
        cognitive_score = (level_hierarchy.index(primary_level) + 1) / len(level_hierarchy)
        
        return {
            "level": primary_level,
            "score": cognitive_score,
            "keyword_matches": level_scores,
            "appropriate_for_level": primary_level in ["understand", "apply", "analyze"]
        }

    def _calculate_clarity_score(self, question: Question) -> Dict[str, Any]:
        """Assess question clarity and identify potential ambiguities"""
        text = question.question_text
        issues = []
        
        # Check for common clarity problems
        if "NOT" in text.upper():
            issues.append("Negative questions can be confusing")
        
        if text.count("?") != 1:
            issues.append("Question should have exactly one question mark")
        
        if len(text.split()) < 5:
            issues.append("Question may be too short/vague")
        
        if len(text.split()) > 30:
            issues.append("Question may be too long/complex")
        
        # Check for ambiguous words
        ambiguous_words = ["some", "many", "often", "usually", "might", "could"]
        found_ambiguous = [word for word in ambiguous_words if word in text.lower()]
        if found_ambiguous:
            issues.append(f"Contains ambiguous words: {', '.join(found_ambiguous)}")
        
        # Check option lengths for balance
        if hasattr(question, 'options') and question.options:
            option_lengths = [len(option.split()) for option in question.options.values()]
            if max(option_lengths) > 2 * min(option_lengths):
                issues.append("Option lengths are unbalanced")
        
        clarity_score = max(0, 1.0 - len(issues) * 0.2)
        
        return {
            "score": clarity_score,
            "issues": issues,
            "word_count": len(text.split()),
            "clear": len(issues) == 0
        }

    def _predict_difficulty(self, question: Question) -> Dict[str, Any]:
        """Predict question difficulty based on various factors"""
        factors = {}
        
        # Readability factor
        if hasattr(question, 'question_text'):
            word_count = len(question.question_text.split())
            factors["complexity"] = min(1.0, word_count / 20)  # Longer = harder
        
        # Distractor quality factor
        if question.distractors:
            avg_closeness = statistics.mean([d.closeness_score for d in question.distractors])
            factors["distractor_difficulty"] = avg_closeness / 10
        
        # Concept complexity (placeholder - could be enhanced with domain knowledge)
        factors["concept_complexity"] = 0.5  # Default middle difficulty
        
        # Combine factors
        predicted_difficulty = statistics.mean(factors.values())
        
        # Map to difficulty levels
        if predicted_difficulty < 0.4:
            level = "EASY"
        elif predicted_difficulty < 0.7:
            level = "MEDIUM"
        else:
            level = "HARD"
        
        return {
            "score": predicted_difficulty,
            "level": level,
            "factors": factors,
            "confidence": 0.7  # Could be improved with more training data
        }

    def _analyze_language_quality(self, question: Question) -> Dict[str, Any]:
        """Analyze language quality and grammatical correctness"""
        text = question.question_text
        issues = []
        
        # Basic grammar checks
        if not text[0].isupper():
            issues.append("Question should start with capital letter")
        
        if not text.rstrip().endswith('?'):
            issues.append("Question should end with question mark")
        
        # Check for common issues
        if "  " in text:  # Double spaces
            issues.append("Contains double spaces")
        
        if any(char in text for char in ['[', ']', '{', '}']):
            issues.append("Contains placeholder brackets")
        
        # Word repetition check
        words = text.lower().split()
        if len(words) != len(set(words)):
            issues.append("Contains repeated words")
        
        quality_score = max(0, 1.0 - len(issues) * 0.25)
        
        return {
            "score": quality_score,
            "issues": issues,
            "grammar_ok": len(issues) == 0
        }

    def _analyze_concept_focus(self, question: Question) -> Dict[str, Any]:
        """Analyze how well the question focuses on key concepts"""
        text = question.question_text.lower()
        
        # Blockchain-specific concepts (could be made configurable)
        blockchain_concepts = [
            "blockchain", "bitcoin", "ethereum", "smart contract", "consensus",
            "mining", "proof of work", "proof of stake", "hash", "merkle tree",
            "wallet", "private key", "public key", "transaction", "block",
            "decentralized", "distributed", "immutable", "cryptography"
        ]
        
        concepts_mentioned = [concept for concept in blockchain_concepts if concept in text]
        
        return {
            "concepts_mentioned": concepts_mentioned,
            "concept_count": len(concepts_mentioned),
            "focused": len(concepts_mentioned) >= 1,
            "primary_concept": concepts_mentioned[0] if concepts_mentioned else "general"
        }

    def _grade_quality(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        else:
            return "D"

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improving question quality"""
        recommendations = []
        
        # Readability recommendations
        readability = analysis.get("readability", {})
        if readability.get("score", 0) < 0.6:
            recommendations.append("Simplify language for better readability")
        
        # Distractor recommendations
        distractor_quality = analysis.get("distractor_quality", {})
        if distractor_quality.get("strong_count", 0) < 2:
            recommendations.append("Add more plausible distractors (closeness score 6+)")
        
        if distractor_quality.get("diversity", 0) < 0.5:
            recommendations.append("Use different misconception types for distractors")
        
        # Clarity recommendations
        clarity = analysis.get("clarity_score", {})
        if clarity.get("score", 0) < 0.7:
            recommendations.append("Improve question clarity by addressing: " + 
                                 ", ".join(clarity.get("issues", [])))
        
        # Cognitive level recommendations
        cognitive = analysis.get("cognitive_level", {})
        if cognitive.get("level") == "remember":
            recommendations.append("Consider raising cognitive level beyond basic recall")
        
        # Language quality recommendations
        language = analysis.get("language_quality", {})
        if not language.get("grammar_ok", True):
            recommendations.append("Fix grammar and formatting issues")
        
        return recommendations

class AssessmentAnalyzer:
    """Analyzes assessment-level quality and provides insights"""
    
    def __init__(self):
        self.quality_analyzer = QuestionQualityAnalyzer()

    def analyze_assessment(self, assessment: AssessmentResponse) -> Dict[str, Any]:
        """
        Comprehensive analysis of an entire assessment
        """
        if not assessment.questions:
            return {"error": "No questions to analyze"}
        
        # Analyze individual questions
        question_analyses = []
        for question in assessment.questions:
            analysis = self.quality_analyzer.analyze_question_quality(question)
            question_analyses.append(analysis)
        
        # Assessment-level metrics
        analysis = {
            "overview": self._generate_overview(assessment, question_analyses),
            "quality_distribution": self._analyze_quality_distribution(question_analyses),
            "difficulty_analysis": self._analyze_difficulty_distribution(assessment.questions),
            "concept_coverage": self._analyze_concept_coverage(assessment.questions),
            "diversity_metrics": self._analyze_diversity(assessment.questions),
            "misconception_analysis": self._analyze_misconception_usage(assessment.questions),
            "recommendations": self._generate_assessment_recommendations(question_analyses),
            "quality_score": self._calculate_assessment_score(question_analyses),
            "detailed_questions": question_analyses
        }
        
        return analysis

    def _generate_overview(self, assessment: AssessmentResponse, question_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate high-level overview of assessment quality"""
        total_questions = len(assessment.questions)
        avg_quality = statistics.mean([q.get("overall_score", 0) for q in question_analyses])
        
        grade_distribution = Counter([q.get("quality_grade", "C") for q in question_analyses])
        high_quality_count = sum(count for grade, count in grade_distribution.items() 
                               if grade in ["A+", "A", "A-", "B+"])
        
        return {
            "total_questions": total_questions,
            "average_quality_score": round(avg_quality, 3),
            "overall_grade": self.quality_analyzer._grade_quality(avg_quality),
            "high_quality_questions": high_quality_count,
            "high_quality_percentage": round((high_quality_count / total_questions) * 100, 1),
            "grade_distribution": dict(grade_distribution),
            "processing_time": getattr(assessment, 'processing_time', 0),
            "source_file": getattr(assessment, 'source_file', 'unknown')
        }

    def _analyze_quality_distribution(self, question_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of quality scores across questions"""
        scores = [q.get("overall_score", 0) for q in question_analyses]
        
        return {
            "mean": round(statistics.mean(scores), 3),
            "median": round(statistics.median(scores), 3),
            "std_dev": round(statistics.stdev(scores) if len(scores) > 1 else 0, 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "quartiles": {
                "q1": round(np.percentile(scores, 25), 3) if ANALYTICS_AVAILABLE else 0,
                "q3": round(np.percentile(scores, 75), 3) if ANALYTICS_AVAILABLE else 0
            }
        }

    def _analyze_difficulty_distribution(self, questions: List[Question]) -> Dict[str, Any]:
        """Analyze difficulty distribution and balance"""
        difficulties = [q.difficulty for q in questions if hasattr(q, 'difficulty')]
        if not difficulties:
            return {"error": "No difficulty information found"}
        
        distribution = Counter(difficulties)
        total = len(difficulties)
        
        # Ideal distribution (configurable)
        ideal_distribution = {"EASY": 0.3, "MEDIUM": 0.5, "HARD": 0.2}
        
        balance_score = 1.0 - sum(
            abs(distribution.get(level, 0) / total - ideal_ratio)
            for level, ideal_ratio in ideal_distribution.items()
        ) / 2
        
        return {
            "distribution": dict(distribution),
            "percentages": {level: round((count / total) * 100, 1) 
                          for level, count in distribution.items()},
            "balance_score": round(balance_score, 3),
            "well_balanced": balance_score > 0.8,
            "ideal_distribution": ideal_distribution
        }

    def _analyze_concept_coverage(self, questions: List[Question]) -> Dict[str, Any]:
        """Analyze which concepts are covered and identify gaps"""
        if not ANALYTICS_AVAILABLE:
            return {"error": "Analytics dependencies not available"}
        
        # Extract concepts from question texts
        question_texts = [q.question_text for q in questions]
        
        # Use TF-IDF to identify key concepts
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(question_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top concepts
            concept_scores = np.sum(tfidf_matrix.toarray(), axis=0)
            top_concepts = sorted(
                zip(feature_names, concept_scores),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "top_concepts": [{"concept": concept, "score": round(score, 3)} 
                               for concept, score in top_concepts],
                "concept_diversity": len(set(feature_names)),
                "coverage_assessment": "good" if len(top_concepts) >= 5 else "limited"
            }
        except Exception as e:
            return {"error": f"Concept analysis failed: {str(e)}"}

    def _analyze_diversity(self, questions: List[Question]) -> Dict[str, Any]:
        """Analyze diversity across multiple dimensions"""
        # Question type diversity
        question_types = [getattr(q, 'question_type', 'multiple_choice') for q in questions]
        type_diversity = len(set(question_types)) / len(question_types) if question_types else 0
        
        # Similarity analysis
        if ANALYTICS_AVAILABLE and len(questions) > 1:
            question_texts = [q.question_text for q in questions]
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(question_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Average similarity (excluding diagonal)
                n = len(questions)
                total_similarity = np.sum(similarity_matrix) - n  # Exclude diagonal
                avg_similarity = total_similarity / (n * (n - 1)) if n > 1 else 0
                
                diversity_score = 1 - avg_similarity
            except:
                diversity_score = 0.5  # Default
        else:
            diversity_score = 0.5
        
        return {
            "type_diversity": round(type_diversity, 3),
            "content_diversity": round(diversity_score, 3),
            "overall_diversity": round((type_diversity + diversity_score) / 2, 3),
            "diversity_assessment": "high" if diversity_score > 0.7 else "medium" if diversity_score > 0.4 else "low"
        }

    def _analyze_misconception_usage(self, questions: List[Question]) -> Dict[str, Any]:
        """Analyze usage of misconception patterns in distractors"""
        all_misconceptions = []
        
        for question in questions:
            if hasattr(question, 'distractors') and question.distractors:
                misconceptions = [d.misconception_type for d in question.distractors]
                all_misconceptions.extend(misconceptions)
        
        if not all_misconceptions:
            return {"error": "No misconception data found"}
        
        misconception_counts = Counter(all_misconceptions)
        total = len(all_misconceptions)
        
        return {
            "total_distractors": total,
            "unique_misconceptions": len(misconception_counts),
            "misconception_distribution": dict(misconception_counts),
            "most_common": misconception_counts.most_common(5),
            "diversity_score": len(misconception_counts) / total if total > 0 else 0
        }

    def _generate_assessment_recommendations(self, question_analyses: List[Dict]) -> List[str]:
        """Generate recommendations for improving the overall assessment"""
        recommendations = []
        
        # Quality recommendations
        low_quality_count = sum(1 for q in question_analyses if q.get("overall_score", 0) < 0.6)
        if low_quality_count > 0:
            recommendations.append(f"Improve {low_quality_count} low-quality questions")
        
        # Difficulty balance
        difficulties = [q.get("estimated_difficulty", {}).get("level", "MEDIUM") 
                       for q in question_analyses]
        difficulty_counts = Counter(difficulties)
        total = len(difficulties)
        
        if difficulty_counts.get("EASY", 0) / total > 0.5:
            recommendations.append("Add more challenging questions")
        elif difficulty_counts.get("HARD", 0) / total > 0.4:
            recommendations.append("Add more easier questions for better balance")
        
        # Cognitive level diversity
        cognitive_levels = [q.get("cognitive_level", {}).get("level", "understand") 
                          for q in question_analyses]
        if len(set(cognitive_levels)) < 3:
            recommendations.append("Increase cognitive level diversity")
        
        # Common quality issues
        common_issues = []
        for q in question_analyses:
            for category in ["clarity_score", "distractor_quality", "language_quality"]:
                issues = q.get(category, {}).get("issues", [])
                common_issues.extend(issues)
        
        issue_counts = Counter(common_issues)
        for issue, count in issue_counts.most_common(3):
            if count > len(question_analyses) * 0.3:  # If >30% of questions have this issue
                recommendations.append(f"Address common issue: {issue}")
        
        return recommendations

    def _calculate_assessment_score(self, question_analyses: List[Dict]) -> float:
        """Calculate overall assessment quality score"""
        if not question_analyses:
            return 0.0
        
        individual_scores = [q.get("overall_score", 0) for q in question_analyses]
        base_score = statistics.mean(individual_scores)
        
        # Bonus for consistency (lower standard deviation)
        consistency_bonus = 0
        if len(individual_scores) > 1:
            std_dev = statistics.stdev(individual_scores)
            consistency_bonus = max(0, (0.2 - std_dev) * 0.5)  # Up to 0.1 bonus
        
        # Bonus for diversity
        diversity_bonus = 0
        if len(set(q.get("cognitive_level", {}).get("level", "understand") 
                  for q in question_analyses)) >= 3:
            diversity_bonus = 0.05
        
        final_score = min(1.0, base_score + consistency_bonus + diversity_bonus)
        return round(final_score, 3)

class PerformanceTracker:
    """Tracks generation performance and stores historical data"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(settings.PROJECT_ROOT / "data" / "analytics.db")
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for performance tracking"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    timestamp TEXT,
                    source_file TEXT,
                    num_questions_requested INTEGER,
                    num_questions_generated INTEGER,
                    generation_time REAL,
                    success_rate REAL,
                    retry_count INTEGER,
                    average_quality_score REAL,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS question_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    question_index INTEGER,
                    quality_score REAL,
                    difficulty TEXT,
                    cognitive_level TEXT,
                    distractor_count INTEGER,
                    strong_distractor_count INTEGER,
                    misconception_types TEXT,
                    processing_time REAL,
                    retry_count INTEGER,
                    FOREIGN KEY (session_id) REFERENCES generation_sessions (id)
                )
            """)

    def log_generation_session(self, task_id: str, session_data: Dict[str, Any]):
        """Log a complete generation session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO generation_sessions (
                    task_id, timestamp, source_file, num_questions_requested,
                    num_questions_generated, generation_time, success_rate,
                    retry_count, average_quality_score, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                datetime.now().isoformat(),
                session_data.get('source_file', ''),
                session_data.get('num_questions_requested', 0),
                session_data.get('num_questions_generated', 0),
                session_data.get('generation_time', 0),
                session_data.get('success_rate', 0),
                session_data.get('retry_count', 0),
                session_data.get('average_quality_score', 0),
                session_data.get('error_message', ''),
                json.dumps(session_data.get('metadata', {}))
            ))

    def log_question_metrics(self, task_id: str, question_metrics: List[Dict[str, Any]]):
        """Log individual question metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get session ID
            cursor = conn.execute(
                "SELECT id FROM generation_sessions WHERE task_id = ?", 
                (task_id,)
            )
            session_row = cursor.fetchone()
            if not session_row:
                return
            
            session_id = session_row[0]
            
            # Insert question metrics
            for i, metrics in enumerate(question_metrics):
                conn.execute("""
                    INSERT INTO question_metrics (
                        session_id, question_index, quality_score, difficulty,
                        cognitive_level, distractor_count, strong_distractor_count,
                        misconception_types, processing_time, retry_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, i,
                    metrics.get('quality_score', 0),
                    metrics.get('difficulty', ''),
                    metrics.get('cognitive_level', ''),
                    metrics.get('distractor_count', 0),
                    metrics.get('strong_distractor_count', 0),
                    json.dumps(metrics.get('misconception_types', [])),
                    metrics.get('processing_time', 0),
                    metrics.get('retry_count', 0)
                ))

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    AVG(success_rate) as avg_success_rate,
                    AVG(generation_time) as avg_generation_time,
                    AVG(average_quality_score) as avg_quality,
                    SUM(num_questions_generated) as total_questions
                FROM generation_sessions 
                WHERE timestamp > ?
            """, (since_date,))
            
            overall_stats = dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))
            
            # Quality trends
            cursor = conn.execute("""
                SELECT DATE(timestamp) as date, AVG(average_quality_score) as quality
                FROM generation_sessions 
                WHERE timestamp > ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (since_date,))
            
            quality_trends = [dict(zip([desc[0] for desc in cursor.description], row)) 
                            for row in cursor.fetchall()]
            
            return {
                "period_days": days,
                "overall_statistics": overall_stats,
                "quality_trends": quality_trends,
                "generated_at": datetime.now().isoformat()
            }

class AnalyticsManager:
    """Main analytics manager that coordinates all analysis components"""
    
    def __init__(self):
        self.question_analyzer = QuestionQualityAnalyzer()
        self.assessment_analyzer = AssessmentAnalyzer()
        self.performance_tracker = PerformanceTracker()

    def analyze_generation_result(self, task_id: str, assessment: AssessmentResponse, 
                                generation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete analysis of a generation result
        """
        # Analyze assessment quality
        assessment_analysis = self.assessment_analyzer.analyze_assessment(assessment)
        
        # Extract metrics for tracking
        question_metrics = []
        for i, question in enumerate(assessment.questions):
            q_analysis = self.question_analyzer.analyze_question_quality(question)
            question_metrics.append({
                'quality_score': q_analysis.get('overall_score', 0),
                'difficulty': getattr(question, 'difficulty', 'MEDIUM'),
                'cognitive_level': q_analysis.get('cognitive_level', {}).get('level', 'understand'),
                'distractor_count': len(getattr(question, 'distractors', [])),
                'strong_distractor_count': q_analysis.get('distractor_quality', {}).get('strong_count', 0),
                'misconception_types': [d.misconception_type for d in getattr(question, 'distractors', [])],
                'processing_time': generation_metadata.get('per_question_time', 0),
                'retry_count': generation_metadata.get('retry_count', 0)
            })
        
        # Log to database
        session_data = {
            'source_file': getattr(assessment, 'source_file', ''),
            'num_questions_requested': generation_metadata.get('num_questions_requested', 0),
            'num_questions_generated': len(assessment.questions),
            'generation_time': getattr(assessment, 'processing_time', 0),
            'success_rate': len(assessment.questions) / generation_metadata.get('num_questions_requested', 1),
            'retry_count': generation_metadata.get('total_retries', 0),
            'average_quality_score': assessment_analysis.get('quality_score', 0),
            'error_message': generation_metadata.get('error_message', ''),
            'metadata': generation_metadata
        }
        
        self.performance_tracker.log_generation_session(task_id, session_data)
        self.performance_tracker.log_question_metrics(task_id, question_metrics)
        
        # Combine all analysis
        complete_analysis = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "assessment_analysis": assessment_analysis,
            "session_performance": session_data,
            "recommendations": self._generate_system_recommendations(assessment_analysis, session_data),
            "quality_alerts": self._check_quality_alerts(assessment_analysis)
        }
        
        return complete_analysis

    def _generate_system_recommendations(self, assessment_analysis: Dict[str, Any], 
                                       session_data: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Performance recommendations
        if session_data.get('success_rate', 1) < 0.8:
            recommendations.append("Consider reducing batch size to improve success rate")
        
        if session_data.get('generation_time', 0) > 60:
            recommendations.append("Generation time is high - consider optimizing prompts")
        
        # Quality recommendations
        avg_quality = assessment_analysis.get('quality_score', 0)
        if avg_quality < 0.6:
            recommendations.append("Overall question quality is low - review generation parameters")
        
        return recommendations

    def _check_quality_alerts(self, assessment_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check for quality issues that need immediate attention"""
        alerts = []
        
        overview = assessment_analysis.get('overview', {})
        if overview.get('high_quality_percentage', 0) < 50:
            alerts.append({
                "level": "warning",
                "message": f"Only {overview.get('high_quality_percentage', 0)}% of questions are high quality"
            })
        
        difficulty_analysis = assessment_analysis.get('difficulty_analysis', {})
        if not difficulty_analysis.get('well_balanced', True):
            alerts.append({
                "level": "info",
                "message": "Difficulty distribution is not well balanced"
            })
        
        return alerts

    def generate_analytics_report(self, task_id: str) -> Dict[str, Any]:
        """Generate comprehensive analytics report for a task"""
        # This would typically load from database and generate full report
        return {
            "task_id": task_id,
            "report_generated": datetime.now().isoformat(),
            "summary": "Comprehensive analytics report",
            "charts_data": self._generate_chart_data(task_id),
            "detailed_metrics": "Available in full analysis"
        }

    def _generate_chart_data(self, task_id: str) -> Dict[str, Any]:
        """Generate data for visualization charts"""
        return {
            "quality_distribution": {"labels": ["A", "B", "C", "D"], "values": [30, 45, 20, 5]},
            "difficulty_distribution": {"labels": ["Easy", "Medium", "Hard"], "values": [30, 50, 20]},
            "misconception_usage": {"labels": ["Type A", "Type B"], "values": [60, 40]}
        }

# Export main analytics instance
analytics_manager = AnalyticsManager()
