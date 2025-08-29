#!/usr/bin/env python3
"""
Question Quality Validation Script
Comprehensive validation of generated assessment questions
"""

import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class QualityIssue:
    """Represents a quality issue found in a question"""
    question_id: int
    severity: str  # 'critical', 'major', 'minor'
    issue_type: str
    description: str
    suggestion: str = ""

class QuestionQualityValidator:
    """Comprehensive question quality validator"""
    
    def __init__(self):
        self.placeholder_patterns = [
            r'\b(option\s+[abcd])\b',
            r'\b(choice\s+[abcd])\b', 
            r'\[.*?\]',
            r'\{.*?\}',
            r'xxx+',
            r'\.\.\.+',
            r'placeholder',
            r'insert\s+text',
            r'answer\s+here'
        ]
        
        self.quality_thresholds = {
            'min_option_length': 3,  # words
            'max_option_length': 30,  # words
            'min_question_length': 5,  # words
            'max_question_length': 50,  # words
            'min_distractors': 2,
            'closeness_threshold': 6  # minimum closeness score
        }
    
    def validate_assessment(self, assessment_data: Dict) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """Validate entire assessment and return issues + metrics"""
        issues = []
        metrics = {
            'total_questions': 0,
            'critical_issues': 0,
            'major_issues': 0,
            'minor_issues': 0,
            'placeholder_violations': 0,
            'length_violations': 0,
            'duplicate_questions': 0,
            'weak_distractors': 0,
            'position_bias': {},
            'difficulty_distribution': {},
            'overall_score': 0
        }
        
        if 'questions' not in assessment_data:
            issues.append(QualityIssue(
                0, 'critical', 'structure', 
                'Assessment missing questions field',
                'Ensure assessment has questions array'
            ))
            return issues, metrics
        
        questions = assessment_data['questions']
        metrics['total_questions'] = len(questions)
        
        # Track for duplicates and distribution analysis
        question_texts = []
        correct_positions = []
        difficulties = []
        
        for i, question in enumerate(questions):
            # Validate individual question
            q_issues = self.validate_question(question, i + 1)
            issues.extend(q_issues)
            
            # Collect data for analysis
            if isinstance(question, dict):
                question_texts.append(question.get('question_text', ''))
                correct_positions.append(question.get('correct_position', 'A'))
                difficulties.append(question.get('difficulty', 'medium'))
        
        # Check for duplicates
        duplicate_issues = self.check_duplicates(question_texts)
        issues.extend(duplicate_issues)
        metrics['duplicate_questions'] = len(duplicate_issues)
        
        # Analyze distributions
        metrics['position_bias'] = self.analyze_position_bias(correct_positions)
        metrics['difficulty_distribution'] = self.analyze_difficulty_distribution(difficulties)
        
        # Count issues by severity
        for issue in issues:
            if issue.severity == 'critical':
                metrics['critical_issues'] += 1
            elif issue.severity == 'major':
                metrics['major_issues'] += 1
            elif issue.severity == 'minor':
                metrics['minor_issues'] += 1
        
        # Calculate overall score
        metrics['overall_score'] = self.calculate_overall_score(metrics)
        
        return issues, metrics
    
    def validate_question(self, question: Dict, question_id: int) -> List[QualityIssue]:
        """Validate a single question"""
        issues = []
        
        if not isinstance(question, dict):
            issues.append(QualityIssue(
                question_id, 'critical', 'structure',
                'Question is not a dictionary object'
            ))
            return issues
        
        # Check required fields
        required_fields = ['question_text', 'options', 'correct_position']
        for field in required_fields:
            if field not in question:
                issues.append(QualityIssue(
                    question_id, 'critical', 'missing_field',
                    f'Missing required field: {field}',
                    f'Add {field} to question structure'
                ))
        
        # Validate question text
        question_text = question.get('question_text', '')
        issues.extend(self.validate_text_field(
            question_text, 'question_text', question_id
        ))
        
        # Validate options
        options = question.get('options', {})
        if not isinstance(options, dict):
            issues.append(QualityIssue(
                question_id, 'critical', 'structure',
                'Options field must be a dictionary'
            ))
        else:
            issues.extend(self.validate_options(options, question_id))
        
        # Validate correct position
        correct_pos = question.get('correct_position', '')
        if correct_pos not in ['A', 'B', 'C', 'D']:
            issues.append(QualityIssue(
                question_id, 'major', 'invalid_position',
                f'Invalid correct_position: {correct_pos}',
                'Use A, B, C, or D for correct_position'
            ))
        
        # Check difficulty
        difficulty = question.get('difficulty', '').lower()
        if difficulty not in ['easy', 'medium', 'hard']:
            issues.append(QualityIssue(
                question_id, 'minor', 'invalid_difficulty',
                f'Invalid difficulty: {difficulty}',
                'Use easy, medium, or hard for difficulty'
            ))
        
        return issues
    
    def validate_text_field(self, text: str, field_name: str, question_id: int) -> List[QualityIssue]:
        """Validate a text field for common issues"""
        issues = []
        
        if not text or not text.strip():
            issues.append(QualityIssue(
                question_id, 'critical', 'empty_field',
                f'{field_name} is empty',
                f'Provide content for {field_name}'
            ))
            return issues
        
        # Check for placeholders
        for pattern in self.placeholder_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(QualityIssue(
                    question_id, 'critical', 'placeholder',
                    f'{field_name} contains placeholder text: {text[:50]}...',
                    'Replace placeholder with actual content'
                ))
        
        # Check length
        word_count = len(text.split())
        min_len = self.quality_thresholds['min_question_length'] if field_name == 'question_text' else self.quality_thresholds['min_option_length']
        max_len = self.quality_thresholds['max_question_length'] if field_name == 'question_text' else self.quality_thresholds['max_option_length']
        
        if word_count < min_len:
            issues.append(QualityIssue(
                question_id, 'major', 'too_short',
                f'{field_name} too short: {word_count} words (min: {min_len})',
                f'Expand {field_name} to at least {min_len} words'
            ))
        elif word_count > max_len:
            issues.append(QualityIssue(
                question_id, 'minor', 'too_long',
                f'{field_name} too long: {word_count} words (max: {max_len})',
                f'Shorten {field_name} to under {max_len} words'
            ))
        
        # Check for common formatting issues
        if text.endswith('?') and field_name != 'question_text':
            issues.append(QualityIssue(
                question_id, 'minor', 'formatting',
                f'{field_name} ends with question mark (should be statement)',
                'Remove question mark from answer option'
            ))
        
        return issues
    
    def validate_options(self, options: Dict, question_id: int) -> List[QualityIssue]:
        """Validate answer options"""
        issues = []
        
        # Check for required option keys
        required_options = ['A', 'B', 'C', 'D']
        for opt in required_options:
            if opt not in options:
                issues.append(QualityIssue(
                    question_id, 'critical', 'missing_option',
                    f'Missing option {opt}',
                    f'Add option {opt} to question'
                ))
        
        # Validate each option
        option_lengths = []
        for opt_key, opt_text in options.items():
            if not isinstance(opt_text, str):
                issues.append(QualityIssue(
                    question_id, 'major', 'invalid_option',
                    f'Option {opt_key} is not a string',
                    f'Ensure option {opt_key} is text'
                ))
                continue
            
            # Validate option text
            opt_issues = self.validate_text_field(opt_text, f'option_{opt_key}', question_id)
            issues.extend(opt_issues)
            
            option_lengths.append(len(opt_text.split()))
        
        # Check option length balance
        if option_lengths and len(option_lengths) >= 3:
            avg_length = sum(option_lengths) / len(option_lengths)
            for i, length in enumerate(option_lengths):
                if abs(length - avg_length) > avg_length * 0.75:  # More than 75% difference
                    issues.append(QualityIssue(
                        question_id, 'minor', 'unbalanced_options',
                        f'Option {required_options[i]} length varies significantly from others',
                        'Balance option lengths for better question quality'
                    ))
        
        return issues
    
    def check_duplicates(self, question_texts: List[str]) -> List[QualityIssue]:
        """Check for duplicate questions"""
        issues = []
        seen = {}
        
        for i, text in enumerate(question_texts):
            text_clean = text.lower().strip()
            if text_clean in seen:
                issues.append(QualityIssue(
                    i + 1, 'major', 'duplicate',
                    f'Duplicate question text (same as question {seen[text_clean]})',
                    'Generate unique question content'
                ))
            else:
                seen[text_clean] = i + 1
        
        return issues
    
    def analyze_position_bias(self, correct_positions: List[str]) -> Dict[str, Any]:
        """Analyze correct answer position distribution"""
        from collections import Counter
        
        position_counts = Counter(correct_positions)
        total = len(correct_positions)
        
        if total == 0:
            return {'bias_detected': False, 'distribution': {}}
        
        distribution = {pos: count/total for pos, count in position_counts.items()}
        
        # Check for bias (more than 40% in one position)
        max_percentage = max(distribution.values()) if distribution else 0
        bias_detected = max_percentage > 0.4
        
        return {
            'bias_detected': bias_detected,
            'max_percentage': max_percentage,
            'distribution': distribution,
            'recommendation': 'Distribute correct answers more evenly' if bias_detected else 'Good distribution'
        }
    
    def analyze_difficulty_distribution(self, difficulties: List[str]) -> Dict[str, Any]:
        """Analyze difficulty level distribution"""
        from collections import Counter
        
        diff_counts = Counter(difficulties)
        total = len(difficulties)
        
        if total == 0:
            return {'distribution': {}}
        
        distribution = {diff: count/total for diff, count in diff_counts.items()}
        
        return {
            'distribution': distribution,
            'total_questions': total,
            'variety_score': len(distribution)  # Number of different difficulty levels
        }
    
    def calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        if metrics['total_questions'] == 0:
            return 0
        
        # Start with 100 and deduct points for issues
        score = 100
        
        # Critical issues are very bad
        score -= metrics['critical_issues'] * 20
        
        # Major issues are significant 
        score -= metrics['major_issues'] * 10
        
        # Minor issues are small deductions
        score -= metrics['minor_issues'] * 2
        
        # Bonus for good practices
        if not metrics.get('position_bias', {}).get('bias_detected', True):
            score += 5
        
        if metrics.get('difficulty_distribution', {}).get('variety_score', 0) >= 2:
            score += 5
        
        return max(0, min(100, score))
    
    def generate_report(self, issues: List[QualityIssue], metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive quality report"""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append("QUESTION QUALITY VALIDATION REPORT")
        report.append("=" * 60)
        
        # Summary
        total_issues = len(issues)
        score = metrics.get('overall_score', 0)
        
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Overall Score: {score:.1f}/100")
        report.append(f"   Total Questions: {metrics.get('total_questions', 0)}")
        report.append(f"   Total Issues: {total_issues}")
        report.append(f"   Critical: {metrics.get('critical_issues', 0)}")
        report.append(f"   Major: {metrics.get('major_issues', 0)}")
        report.append(f"   Minor: {metrics.get('minor_issues', 0)}")
        
        # Grade
        if score >= 90:
            grade = "🏆 EXCELLENT"
        elif score >= 80:
            grade = "✅ GOOD"
        elif score >= 70:
            grade = "⚠️  FAIR"
        elif score >= 60:
            grade = "❌ POOR"
        else:
            grade = "💥 CRITICAL"
        
        report.append(f"\n🎯 GRADE: {grade}")
        
        # Detailed issues
        if issues:
            report.append(f"\n🔍 DETAILED ISSUES:")
            
            # Group by severity
            critical = [i for i in issues if i.severity == 'critical']
            major = [i for i in issues if i.severity == 'major']
            minor = [i for i in issues if i.severity == 'minor']
            
            for severity_group, title in [(critical, 'CRITICAL'), (major, 'MAJOR'), (minor, 'MINOR')]:
                if severity_group:
                    report.append(f"\n   {title} ISSUES ({len(severity_group)}):")
                    for issue in severity_group[:10]:  # Show first 10
                        report.append(f"   • Q{issue.question_id}: {issue.description}")
                        if issue.suggestion:
                            report.append(f"     → {issue.suggestion}")
                    
                    if len(severity_group) > 10:
                        report.append(f"     ... and {len(severity_group) - 10} more {title.lower()} issues")
        
        # Position bias analysis
        pos_bias = metrics.get('position_bias', {})
        if pos_bias.get('bias_detected'):
            report.append(f"\n⚠️  POSITION BIAS DETECTED:")
            report.append(f"   Max percentage in one position: {pos_bias.get('max_percentage', 0)*100:.1f}%")
            report.append(f"   Recommendation: {pos_bias.get('recommendation', '')}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if metrics.get('critical_issues', 0) > 0:
            report.append("   1. Fix all critical issues immediately - these prevent proper functionality")
        if metrics.get('major_issues', 0) > 0:
            report.append("   2. Address major issues - these significantly impact quality")
        if pos_bias.get('bias_detected'):
            report.append("   3. Randomize correct answer positions more effectively")
        if metrics.get('duplicate_questions', 0) > 0:
            report.append("   4. Ensure all questions are unique")
        
        report.append("\nFor help fixing these issues, refer to the manual testing guide.")
        
        return "\n".join(report)

def validate_json_file(file_path: str) -> None:
    """Validate questions from a JSON file"""
    validator = QuestionQualityValidator()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        issues, metrics = validator.validate_assessment(data)
        report = validator.generate_report(issues, metrics)
        
        print(report)
        
        # Save report
        report_path = Path(file_path).with_suffix('.quality_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error validating file {file_path}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python quality_validator.py <assessment_file.json>")
        sys.exit(1)
    
    validate_json_file(sys.argv[1])
