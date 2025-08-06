import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import PyPDF2
from docx import Document
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume Analyzer API", version="1.0.0")

# --- 1. Pydantic Schemas ---

class CompanyDetail(BaseModel):
    """Structured details for a single company."""
    company_name: str = Field(description="The name of the company.")
    legitimacy_status: str = Field(description="Legitimacy assessment (e.g., 'Legitimate', 'Suspicious', 'Unknown').")
    payment_status: str = Field(description="Payment structure (e.g., 'Paid', 'Unpaid', 'Student Paid Fee', 'Unknown').")
    key_findings: str = Field(description="Brief summary of evidence and findings.")
    reputation_score: int = Field(description="Score from 1-10 indicating company reputation.")

class JobMatch(BaseModel):
    """Resume-to-job-description match analysis."""
    match_score: float = Field(description="Score out of 100 for resume-job match.")
    summary: str = Field(description="Summary explaining the score and key points.")

class UnifiedReport(BaseModel):
    """Complete analysis report."""
    job_match_analysis: JobMatch
    company_verification: List[CompanyDetail]

class AnalysisResponse(BaseModel):
    """FastAPI response model."""
    company_verification: List[CompanyDetail]
    job_match_analysis: JobMatch

class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    resume_text: str = Field(..., description="Resume text content")
    job_description_text: str = Field(..., description="Job description text content")


# --- 2. File Processing Utilities ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

def process_resume_file(file_path: str) -> str:
    """Process a resume file (PDF or DOCX) and extract text."""
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return f"Unsupported file type: {file_path}"

def extract_text_from_file_bytes(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded file bytes (PDF or DOCX)."""
    try:
        if filename.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                with open(tmp_file.name, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                os.unlink(tmp_file.name)
                return text.strip()
                
        elif filename.lower().endswith('.docx'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                doc = Document(tmp_file.name)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                os.unlink(tmp_file.name)
                return text.strip()
        else:
            return f"Unsupported file type: {filename}"
    except Exception as e:
        return f"Error extracting text from {filename}: {str(e)}"


# --- 3. Core Analysis Engine ---

class ResumeAnalyzer:
    """
    Main class that handles both job matching and company verification.
    Optimized for speed and accuracy.
    """
    
    def __init__(self):
        """Initialize the analyzer with optimized settings."""
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is missing in the environment.")
            
        self.llm = ChatGroq(
            model="llama3-8b-8192",  # Fast and reliable model
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
        
        self.search_tool = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Complete resume analysis including company verification and job matching.
        """
        try:
            print("🔄 Starting company verification...")
            company_verification = self._verify_companies(resume_text)
            
            print("🔄 Starting job match analysis...")
            job_match = self._evaluate_job_match(resume_text, job_description)
            
            return {
                "company_verification": company_verification,
                "job_match_analysis": job_match
            }
            
        except Exception as e:
            return {
                "error": "Analysis failed",
                "details": str(e),
                "company_verification": [],
                "job_match_analysis": {"match_score": 0, "summary": "Error occurred"}
            }

    def _evaluate_job_match(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Evaluate how well the resume matches the job description and provide comprehensive assessment."""
        prompt = f"""
        You are an expert HR analyst. Evaluate how well this resume matches the given job description and provide a concise assessment.

        Job Description:
        {job_description}

        Resume:
        {resume_text}

        Analyze the resume against the job requirements and provide:
        1. A numerical score from 0 to 100 (where 100 is a perfect match)
        2. A concise summary in 2-3 lines that includes:
           - Key strengths of the candidate that match the job requirements
           - Main weaknesses or gaps compared to the job requirements
           - Overall assessment of fit

        Consider:
        - Required skills and technologies alignment
        - Years of experience vs requirements
        - Educational background relevance
        - Relevant projects and achievements
        - Specific gaps or missing requirements

        Respond in exactly this format:
        Score: [your numerical score from 0-100]
        Summary: [2-3 line concise analysis covering strengths, weaknesses, and overall fit]
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            output = response.content.strip()
            
            # Let the agent parse its own response
            parse_prompt = f"""
            Extract the score and summary from this analysis:

            {output}

            Return in exactly this format:
            SCORE: [number only]
            SUMMARY: [complete summary text only - ensure it's 2-3 lines covering strengths and weaknesses]
            """
            
            parse_response = self.llm.invoke([HumanMessage(content=parse_prompt)])
            parsed_output = parse_response.content.strip()
            
            # Extract score and summary from agent's parsed response
            score = 0
            summary = "Unable to evaluate"
            
            for line in parsed_output.split('\n'):
                line = line.strip()
                if line.startswith('SCORE:'):
                    try:
                        score_text = line.replace('SCORE:', '').strip()
                        score = float(score_text)
                    except (ValueError, TypeError):
                        score = 0
                elif line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
            
            # If parsing failed, ask agent to provide components individually
            if score == 0 and summary == "Unable to evaluate":
                score_prompt = f"From this analysis, what is the numerical score (0-100)? Return only the number: {output}"
                score_response = self.llm.invoke([HumanMessage(content=score_prompt)])
                try:
                    score = float(score_response.content.strip())
                except:
                    score = 50
                
                summary_prompt = f"From this analysis, provide a concise 2-3 line summary covering candidate strengths, weaknesses, and overall fit: {output}"
                summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                summary = summary_response.content.strip()
            
            return {
                "match_score": score,
                "summary": summary
            }
            
        except Exception as e:
            return {
                "match_score": 0,
                "summary": f"Error evaluating job match: {str(e)}"
            }

    def _extract_companies(self, resume_text: str) -> List[str]:
        """
        Extract company names ONLY from work experience and internships using pure agent approach.
        No regex or hardcoded logic - only LLM analysis.
        """
        prompt = f"""
        You are an expert HR analyst. Analyze this resume text and extract ONLY company names where the person worked or did internships.

        STRICT RULES:
        1. Extract companies ONLY from work experience and internship sections
        2. Do NOT include educational institutions (universities, colleges, institutes, schools)
        3. Do NOT include API services, tools, or technologies (like Gemini, Meta, Brevo, Cloudinary)
        4. Do NOT include workshop providers, training centers, or certificate programs
        5. Do NOT include URLs, links, or web addresses (like "LINK", "link", "http://", etc.)
        6. Do NOT include project names, repository names, or GitHub links
        7. Look for actual employment relationships with real company entities

        Resume Text:
        {resume_text}

        Analyze the resume carefully and return only the actual company names where this person was employed or did internships. Ignore any URLs, links, or web addresses that might appear in project descriptions.

        Return each company name on a separate line with no additional text, numbers, or explanations.

        If you find companies, list them like this:
        CompanyName1
        CompanyName2
        CompanyName3

        If no companies are found, return: NO_COMPANIES_FOUND
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            companies_text = response.content.strip()
            
            if not companies_text or companies_text == "NO_COMPANIES_FOUND":
                print("🏢 No companies found by agent")
                return []
            
            # Split by lines and clean
            lines = companies_text.split('\n')
            companies = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Remove any numbering or bullet points that might be added
                line = line.replace('1. ', '').replace('2. ', '').replace('3. ', '')
                line = line.replace('- ', '').replace('• ', '').replace('* ', '')
                line = line.strip()
                
                # Skip if it's still empty or too short
                if len(line) < 3:
                    continue
                
                # Skip obvious agent artifacts and URLs/links
                if any(phrase in line.lower() for phrase in [
                    'no companies', 'found', 'based on', 'analysis', 'resume',
                    'link', 'http', 'www', '.com', '.org', '.net', 'github',
                    'url', 'website', 'repository', 'repo'
                ]):
                    continue
                
                companies.append(line)
            
            # Remove duplicates while preserving order
            unique_companies = []
            for company in companies:
                if company not in unique_companies:
                    unique_companies.append(company)
            
            print(f"🏢 Extracted companies (agent): {unique_companies}")
            return unique_companies[:3]  # Limit to 3 companies for performance
            
        except Exception as e:
            print(f"❌ Error extracting companies: {e}")
            return []

    def _verify_companies(self, resume_text: str) -> List[Dict[str, Any]]:
        """Verify the legitimacy of extracted companies using pure agent analysis."""
        companies = self._extract_companies(resume_text)
        verification_results = []
        
        for company in companies:
            try:
                print(f"🔍 Verifying: {company}")
                
                # Get search results for company information
                try:
                    search_results = self.search_tool.invoke(f"{company} company reviews legitimacy employee payment")
                    search_text = str(search_results) if search_results else "No search results available"
                except Exception as search_error:
                    print(f"⚠️ Search failed for {company}: {search_error}")
                    search_text = "Search unavailable"
                
                # Let the agent analyze the company comprehensively
                verification_prompt = f"""
                You are an expert company analyst. Analyze the company '{company}' based on the following information and provide a comprehensive verification report.

                Available Information:
                {search_text}

                Analyze and determine:
                1. Legitimacy Status: Is this a legitimate company? (Legitimate/Suspicious/Unknown)
                2. Payment Status: How does this company handle employee/intern payments? (Paid/Unpaid/Student Paid Fee/Unknown)
                3. Key Findings: What are the most important findings about this company?
                4. Reputation Score: Rate the company's reputation from 1-10 (1=very poor, 10=excellent)

                Provide your analysis in exactly this format:
                STATUS: [Legitimate/Suspicious/Unknown]
                PAYMENT: [Paid/Unpaid/Student Paid Fee/Unknown]
                FINDINGS: [One sentence summary of key findings]
                SCORE: [number from 1-10]
                """
                
                response = self.llm.invoke([HumanMessage(content=verification_prompt)])
                result_text = response.content.strip()
                
                # Let the agent parse its own response
                parse_prompt = f"""
                Extract the specific information from this verification report:

                {result_text}

                Return in exactly this format with only the requested values:
                STATUS: [status value only]
                PAYMENT: [payment value only]  
                FINDINGS: [findings text only]
                SCORE: [score number only]
                """
                
                parse_response = self.llm.invoke([HumanMessage(content=parse_prompt)])
                parsed_result = parse_response.content.strip()
                
                # Extract values from agent's parsed response
                status = "Unknown"
                payment = "Unknown"
                summary = "Limited information available"
                score = 5
                
                for line in parsed_result.split('\n'):
                    line = line.strip()
                    if line.startswith('STATUS:'):
                        status = line.replace('STATUS:', '').strip()
                    elif line.startswith('PAYMENT:'):
                        payment = line.replace('PAYMENT:', '').strip()
                    elif line.startswith('FINDINGS:'):
                        summary = line.replace('FINDINGS:', '').strip()
                    elif line.startswith('SCORE:'):
                        try:
                            score = int(line.replace('SCORE:', '').strip())
                        except (ValueError, TypeError):
                            score = 5
                
                # If parsing failed, ask agent for individual components
                if status == "Unknown" and payment == "Unknown" and summary == "Limited information available":
                    status_prompt = f"From this analysis of {company}, what is the legitimacy status? Answer only: Legitimate, Suspicious, or Unknown: {result_text}"
                    status_response = self.llm.invoke([HumanMessage(content=status_prompt)])
                    status = status_response.content.strip()
                    
                    payment_prompt = f"From this analysis of {company}, what is the payment status? Answer only: Paid, Unpaid, Student Paid Fee, or Unknown: {result_text}"
                    payment_response = self.llm.invoke([HumanMessage(content=payment_prompt)])
                    payment = payment_response.content.strip()
                    
                    findings_prompt = f"From this analysis of {company}, provide a one-sentence summary of key findings: {result_text}"
                    findings_response = self.llm.invoke([HumanMessage(content=findings_prompt)])
                    summary = findings_response.content.strip()
                    
                    score_prompt = f"From this analysis of {company}, what reputation score (1-10) would you give? Answer only with the number: {result_text}"
                    score_response = self.llm.invoke([HumanMessage(content=score_prompt)])
                    try:
                        score = int(score_response.content.strip())
                    except:
                        score = 5
                
                verification_results.append({
                    "company_name": company,
                    "legitimacy_status": status,
                    "payment_status": payment,
                    "key_findings": summary,
                    "reputation_score": score
                })
                
                print(f"✅ Verified: {company} - {status}")
                
            except Exception as e:
                print(f"❌ Error verifying {company}: {e}")
                verification_results.append({
                    "company_name": company,
                    "legitimacy_status": "Error",
                    "payment_status": "Unknown",
                    "key_findings": f"Verification failed: {str(e)}",
                    "reputation_score": 0
                })
        
        return verification_results

# --- 4. FastAPI Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Resume Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Upload resume and job description for analysis",
            "/analyze-text": "POST - Analyze text directly",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not groq_key:
            return {"status": "unhealthy", "message": "GROQ_API_KEY not configured"}
        if not tavily_key:
            return {"status": "unhealthy", "message": "TAVILY_API_KEY not configured"}
        
        return {"status": "healthy", "message": "All systems operational"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Health check failed: {str(e)}"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_files(
    resume_file: UploadFile = File(...),
    job_description_file: UploadFile = File(...)
):
    """Analyze resume and job description files."""
    try:
        # Validate file types
        allowed_extensions = ['.pdf', '.docx']
        
        if not any(resume_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Resume file must be PDF or DOCX")
        
        if not any(job_description_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Job description file must be PDF or DOCX")
        
        # Read file contents
        resume_content = await resume_file.read()
        job_description_content = await job_description_file.read()
        
        # Extract text
        resume_text = extract_text_from_file_bytes(resume_content, resume_file.filename)
        job_description_text = extract_text_from_file_bytes(job_description_content, job_description_file.filename)
        
        if resume_text.startswith("Error"):
            raise HTTPException(status_code=400, detail=f"Resume processing failed: {resume_text}")
        
        if job_description_text.startswith("Error"):
            raise HTTPException(status_code=400, detail=f"Job description processing failed: {job_description_text}")
        
        # Run analysis
        analyzer = ResumeAnalyzer()
        result = analyzer.analyze_resume(resume_text, job_description_text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["details"])
        
        return AnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze resume and job description text."""
    try:
        if len(request.resume_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Resume text too short")
        
        if len(request.job_description_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Job description text too short")
        
        # Run analysis
        analyzer = ResumeAnalyzer()
        result = analyzer.analyze_resume(request.resume_text, request.job_description_text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["details"])
        
        return AnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- 5. Main Application Functions ---

def run_complete_analysis(resume_path: str, job_description_path: str) -> Dict[str, Any]:
    """Run complete analysis on resume and job description files."""
    print(f"📄 Processing resume file: {resume_path}")
    print(f"📄 Processing job description file: {job_description_path}")
    
    # Extract text from resume file
    resume_text = process_resume_file(resume_path)
    
    if resume_text.startswith("Error"):
        return {"error": "Resume file processing failed", "details": resume_text}
    
    print(f"✅ Extracted {len(resume_text)} characters from resume file")
    
    # Extract text from job description file
    job_description_text = process_resume_file(job_description_path)
    
    if job_description_text.startswith("Error"):
        return {"error": "Job description file processing failed", "details": job_description_text}
    
    print(f"✅ Extracted {len(job_description_text)} characters from job description file")
    
    # Run analysis
    analyzer = ResumeAnalyzer()
    result = analyzer.analyze_resume(resume_text, job_description_text)
    
    return result

def run_text_analysis(resume_text: str, job_description_text: str) -> Dict[str, Any]:
    """Run analysis on resume and job description text directly."""
    print("📝 Processing resume and job description text...")
    
    analyzer = ResumeAnalyzer()
    result = analyzer.analyze_resume(resume_text, job_description_text)
    
    return result

# --- 6. Main Execution Block ---

if __name__ == "__main__":
    import sys
    
    # Check if we should run FastAPI or the original script
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run FastAPI server
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Run original script logic
        RESUME_FILE_PATH = "/Users/apple/Desktop/employers/Resume.pdf"
        JOB_DESCRIPTION_FILEPATH = "/Users/apple/Desktop/employers/Untitled document (1).pdf"
        
        print("🚀 Initializing the Optimized Resume Analyzer...")
        print("-" * 60)
        
        try:
            print(f"📁 Processing files:")
            print(f"   Resume: {RESUME_FILE_PATH}")
            print(f"   Job Description: {JOB_DESCRIPTION_FILEPATH}")
            
            final_report = run_complete_analysis(RESUME_FILE_PATH, JOB_DESCRIPTION_FILEPATH)
            
            print("✅ Analysis Complete!")
            print("-" * 60)
            print("📊 Final Report:")
            print(json.dumps(final_report, indent=4))
            
            # Save to file
            output_file = "resume_analysis_report.json"
            with open(output_file, 'w') as f:
                json.dump(final_report, f, indent=4)
            print(f"\n💾 Report saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please check your API keys and file paths.")