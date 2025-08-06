#!/usr/bin/env python3
"""
Resume Analyzer with ATS Scoring
Analyzes PDF resumes using MoonshotAI and provides comprehensive feedback
"""

import json
import os
import sys
from pathlib import Path
import PyPDF2
import requests
from typing import Dict, List, Any
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MoonshotAI:
    """Custom LLM wrapper for MoonshotAI API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        self.model_name = "moonshotai/kimi-k2-instruct"
        self.base_url = "https://api.groq.com/openai/v1"
        
        if not self.api_key:
            raise ValueError("MoonshotAI API key is required")
    
    def _call(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling MoonshotAI: {str(e)}"

class ResumeAnalyzer:
    """Main resume analyzer class"""
    
    def __init__(self, api_key: str = None):
        self.llm = MoonshotAI(api_key=api_key)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def calculate_ats_score(self, resume_text: str) -> Dict[str, Any]:
        """Calculate ATS score using AI analysis"""
        prompt = f"""
        Analyze this resume text for ATS (Applicant Tracking System) compatibility and provide a comprehensive score.
        
        Resume Text:
        {resume_text}
        
        Please analyze and return a JSON response with the following structure:
        {{
            "ats_score": <number between 0-100>,
            "ats_factors": {{
                "formatting_score": <0-25>,
                "keyword_density": <0-25>,
                "section_organization": <0-25>,
                "readability": <0-25>
            }},
            "ats_recommendations": [
                "specific recommendation 1",
                "specific recommendation 2"
            ]
        }}
        
        Consider factors like:
        - Standard section headers (Experience, Education, Skills)
        - Keyword relevance and density
        - Formatting simplicity
        - Contact information completeness
        - Quantifiable achievements
        """
        
        response = self.llm._call(prompt)
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"ats_score": 0, "error": "Could not parse ATS analysis"}
        except:
            return {"ats_score": 0, "error": "Failed to analyze ATS score"}  
  
    def analyze_keywords(self, resume_text: str) -> Dict[str, Any]:
        """Analyze keywords and their relevance"""
        prompt = f"""
        Analyze the keywords in this resume and provide detailed keyword analysis.
        
        Resume Text:
        {resume_text}
        
        Return JSON response:
        {{
            "keyword_analysis": {{
                "technical_skills": ["skill1", "skill2"],
                "soft_skills": ["skill1", "skill2"],
                "industry_keywords": ["keyword1", "keyword2"],
                "missing_keywords": ["missing1", "missing2"],
                "keyword_density_score": <0-100>
            }},
            "keyword_recommendations": [
                "Add more industry-specific keywords",
                "Include relevant technical certifications"
            ]
        }}
        
        Focus on:
        - Technical skills and tools
        - Industry-specific terminology
        - Soft skills and competencies
        - Missing high-impact keywords
        - Keyword frequency and placement
        """
        
        response = self.llm._call(prompt)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"keyword_analysis": {}, "error": "Could not parse keyword analysis"}
        except:
            return {"keyword_analysis": {}, "error": "Failed to analyze keywords"}
    
    def analyze_action_verbs(self, resume_text: str) -> Dict[str, Any]:
        """Analyze action verbs and their strength"""
        prompt = f"""
        Analyze the action verbs used in this resume and evaluate their strength and impact.
        
        Resume Text:
        {resume_text}
        
        Return JSON response:
        {{
            "action_verb_analysis": {{
                "strong_verbs": ["achieved", "implemented", "optimized"],
                "weak_verbs": ["responsible for", "worked on", "helped"],
                "verb_strength_score": <0-100>,
                "verb_variety_score": <0-100>
            }},
            "verb_recommendations": [
                "Replace 'responsible for' with 'managed' or 'led'",
                "Use more quantifiable action verbs"
            ]
        }}
        
        Evaluate:
        - Strength and impact of action verbs
        - Variety and diversity of verbs used
        - Passive vs active voice usage
        - Industry-appropriate terminology
        """
        
        response = self.llm._call(prompt)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"action_verb_analysis": {}, "error": "Could not parse action verb analysis"}
        except:
            return {"action_verb_analysis": {}, "error": "Failed to analyze action verbs"}
    
    def analyze_strengths_weaknesses(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume strengths and weaknesses"""
        prompt = f"""
        Conduct a comprehensive analysis of this resume's strengths and weaknesses.
        
        Resume Text:
        {resume_text}
        
        Return JSON response:
        {{
            "strengths": [
                "Clear quantifiable achievements",
                "Strong technical skill set",
                "Relevant work experience"
            ],
            "weaknesses": [
                "Missing contact information",
                "Lack of quantified results",
                "Poor formatting consistency"
            ],
            "overall_impression": "Brief overall assessment",
            "competitiveness_score": <0-100>
        }}
        
        Analyze:
        - Content quality and relevance
        - Achievement quantification
        - Professional presentation
        - Completeness of information
        - Market competitiveness
        """
        
        response = self.llm._call(prompt)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"strengths": [], "weaknesses": [], "error": "Could not parse strengths/weaknesses"}
        except:
            return {"strengths": [], "weaknesses": [], "error": "Failed to analyze strengths/weaknesses"}    

    def get_improvement_recommendations(self, resume_text: str, analysis_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive improvement recommendations"""
        prompt = f"""
        Based on this resume and analysis results, provide detailed improvement recommendations.
        
        Resume Text:
        {resume_text}
        
        Analysis Results:
        {json.dumps(analysis_results, indent=2)}
        
        Return JSON response:
        {{
            "priority_improvements": [
                {{
                    "category": "Content",
                    "issue": "Missing quantified achievements",
                    "recommendation": "Add specific metrics and numbers to demonstrate impact",
                    "priority": "High",
                    "example": "Increased sales by 25% over 6 months"
                }}
            ],
            "formatting_improvements": [
                "Use consistent bullet points",
                "Ensure proper section headers"
            ],
            "content_improvements": [
                "Add more industry-specific keywords",
                "Include relevant certifications"
            ],
            "overall_strategy": "Brief strategic advice for resume improvement"
        }}
        
        Focus on:
        - High-impact changes
        - ATS optimization
        - Content enhancement
        - Professional presentation
        - Market positioning
        """
        
        response = self.llm._call(prompt)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"priority_improvements": [], "error": "Could not parse recommendations"}
        except:
            return {"priority_improvements": [], "error": "Failed to generate recommendations"}
    
    def analyze_resume(self, pdf_path: str) -> Dict[str, Any]:
        """Main method to analyze resume comprehensively"""
        try:
            # Extract text from PDF
            print("📄 Extracting text from PDF...")
            resume_text = self.extract_text_from_pdf(pdf_path)
            
            if "Error" in resume_text:
                return {"error": resume_text}
            
            print("🔍 Analyzing resume components...")
            
            # Perform all analyses
            ats_analysis = self.calculate_ats_score(resume_text)
            keyword_analysis = self.analyze_keywords(resume_text)
            action_verb_analysis = self.analyze_action_verbs(resume_text)
            strength_weakness_analysis = self.analyze_strengths_weaknesses(resume_text)
            
            # Combine all results
            combined_analysis = {
                "ats_analysis": ats_analysis,
                "keyword_analysis": keyword_analysis,
                "action_verb_analysis": action_verb_analysis,
                "strength_weakness_analysis": strength_weakness_analysis
            }
            
            # Get improvement recommendations
            print("💡 Generating improvement recommendations...")
            recommendations = self.get_improvement_recommendations(resume_text, combined_analysis)
            
            # Compile final results
            final_results = {
                "resume_analysis": {
                    "file_path": pdf_path,
                    "analysis_timestamp": "2025-01-08",
                    "ats_score": ats_analysis.get("ats_score", 0),
                    "ats_details": ats_analysis,
                    "keyword_analysis": keyword_analysis,
                    "action_verb_analysis": action_verb_analysis,
                    "strengths_weaknesses": strength_weakness_analysis,
                    "improvement_recommendations": recommendations,
                    "overall_grade": self.calculate_overall_grade(combined_analysis)
                }
            }
            
            return final_results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def calculate_overall_grade(self, analysis: Dict) -> str:
        """Calculate overall resume grade"""
        try:
            ats_score = analysis.get("ats_analysis", {}).get("ats_score", 0)
            keyword_score = analysis.get("keyword_analysis", {}).get("keyword_analysis", {}).get("keyword_density_score", 0)
            verb_score = analysis.get("action_verb_analysis", {}).get("action_verb_analysis", {}).get("verb_strength_score", 0)
            competitiveness = analysis.get("strength_weakness_analysis", {}).get("competitiveness_score", 0)
            
            avg_score = (ats_score + keyword_score + verb_score + competitiveness) / 4
            
            if avg_score >= 90:
                return "A+ (Excellent)"
            elif avg_score >= 80:
                return "A (Very Good)"
            elif avg_score >= 70:
                return "B (Good)"
            elif avg_score >= 60:
                return "C (Average)"
            else:
                return "D (Needs Improvement)"
        except:
            return "Unable to calculate grade"

def main():
    """Main function to run the resume analyzer"""
    
    # PUT YOUR PDF PATH HERE - UPDATE THIS LINE
    pdf_path = "/Users/suduyr/Downloads/IBM/Y-R-RITTESH-FlowCV-Resume-20250711 (1).pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found - {pdf_path}")
        print("Please update the pdf_path variable in the main() function")
        sys.exit(1)
    
    # Get API key from environment
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("❌ Error: MOONSHOT_API_KEY environment variable not set")
        print("Please set your API key in the .env file")
        sys.exit(1)
    
    try:
        print("🚀 Starting Resume Analysis...")
        print(f"📁 Analyzing: {pdf_path}")
        print("-" * 50)
        
        # Initialize analyzer
        analyzer = ResumeAnalyzer(api_key=api_key)
        
        # Analyze resume
        results = analyzer.analyze_resume(pdf_path)
        
        # Output results as JSON
        print("\n" + "="*50)
        print("📊 RESUME ANALYSIS RESULTS")
        print("="*50)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        if "error" not in results:
            print("\n✅ Analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()