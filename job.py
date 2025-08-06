# Convert all the responses to json
import os
import re
from typing import List, Dict, Any
from pydantic import BaseModel
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import json

load_dotenv()

class CompanyLegitimacyChecker:
    """
    Simple tool for checking company legitimacy only.
    """
    
    def __init__(self):
        """Initialize the company legitimacy checker."""
        self.llm = ChatGroq(
            model="llama3-8b-8192", 
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
        
        self.search_tool = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    
    def check_company_legitimacy(self, company_name: str, job_role: str = None) -> Dict[str, Any]:
        """Check if a company is legitimate and return comprehensive assessment."""
        try:
            # Search for company information
            search_query = f"{company_name} company legitimacy reviews scam check reputation employer review growth revenue scale size"
            search_results = self.search_tool.invoke(search_query)
            
            # Search for job role specific information if provided
            job_role_info = ""
            if job_role:
                job_search_query = f"{company_name} {job_role} employee reviews salary benefits work culture"
                job_search_results = self.search_tool.invoke(job_search_query)
                job_role_info = f"\n\nJob Role Specific Information for '{job_role}':\n{job_search_results}"
            
            verification_prompt = f"""
            Based on these search results about '{company_name}', provide a comprehensive assessment in JSON format:
            
            Search Results:
            {search_results}{job_role_info}
            
            Provide the following information in JSON format:
            {{
                "one_line_summary": "Brief one-line description of what the company does",
                "legitimacy_status": "Legitimate/Suspicious/Scam/Insufficient Data",
                "employer_review": "Overall rating and employee feedback summary",
                "growth": "Company growth status and trajectory",
                "revenue": "Revenue information or financial status",
                "scale": "Company size, number of employees, market presence",
                "job_role_assessment": "Specific assessment for the {job_role if job_role else 'requested'} role (if applicable)",
                "red_flags": "Any warning signs or concerns",
                "positive_indicators": "Good signs about the company"
            }}
            
            Ensure the response is valid JSON format. If information is not available, use "Not available" or "Insufficient data".
            """
            
            response = self.llm.invoke(verification_prompt)
            assessment_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON from the response
            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', assessment_text)
                if json_match:
                    assessment_json = json.loads(json_match.group(0))
                else:
                    # If no JSON found, create structured response
                    assessment_json = {
                        "one_line_summary": "Unable to extract structured data",
                        "legitimacy_status": "Insufficient Data",
                        "employer_review": assessment_text[:200] + "...",
                        "growth": "Not available",
                        "revenue": "Not available", 
                        "scale": "Not available",
                        "job_role_assessment": "Not available" if not job_role else f"No specific data for {job_role}",
                        "red_flags": "Unable to determine",
                        "positive_indicators": "Unable to determine"
                    }
            except json.JSONDecodeError:
                assessment_json = {
                    "one_line_summary": "Data parsing error",
                    "legitimacy_status": "Insufficient Data",
                    "employer_review": assessment_text[:200] + "...",
                    "growth": "Not available",
                    "revenue": "Not available",
                    "scale": "Not available", 
                    "job_role_assessment": "Not available" if not job_role else f"No specific data for {job_role}",
                    "red_flags": "Unable to determine",
                    "positive_indicators": "Unable to determine"
                }
            
            return {
                "company_name": company_name,
                "job_role": job_role,
                "assessment": assessment_json,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "company_name": company_name,
                "job_role": job_role,
                "assessment": f"Error checking legitimacy: {str(e)}",
                "status": "error"
            }

# Pydantic input schema
class CompanyInput(BaseModel):
    company_name: str

def check_company(company_name: str, job_role: str = None) -> Dict[str, Any]:
    """Check a single company's legitimacy."""
    checker = CompanyLegitimacyChecker()
    result = checker.check_company_legitimacy(company_name, job_role)
    
    # Save result to JSON file
    filename_parts = [company_name.replace(' ', '_').replace('/', '_')]
    if job_role:
        filename_parts.append(job_role.replace(' ', '_').replace('/', '_'))
    output_filename = f"{'_'.join(filename_parts)}_legitimacy_check.json"
    
    try:
        with open(output_filename, "w") as f:
            json.dump(result, f, indent=4)
        result["output_file"] = output_filename
    except Exception as e:
        result["file_error"] = f"Could not save to file: {str(e)}"
    
    return result

# --- Test section ---
if __name__ == "__main__":
    print("🔍 Company Legitimacy Checker")
    print("=" * 40)
    
    # Get company name and job role from user
    company_name = input("Enter company name to check: ").strip()
    job_role = input("Enter job role (optional, press Enter to skip): ").strip()
    
    if not job_role:
        job_role = None
    
    if company_name:
        if job_role:
            print(f"\nChecking legitimacy of: {company_name} for role: {job_role}")
        else:
            print(f"\nChecking legitimacy of: {company_name}")
        print("-" * 50)
        
        result = check_company(company_name, job_role)
        
        print(f"\n📊 Company Legitimacy Check Results:")
        print("=" * 50)
        print(f"Company: {result['company_name']}")
        if result.get('job_role'):
            print(f"Job Role: {result['job_role']}")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success' and isinstance(result.get('assessment'), dict):
            assessment = result['assessment']
            print(f"\n📝 Assessment Details:")
            print("-" * 30)
            print(f"Summary: {assessment.get('one_line_summary', 'N/A')}")
            print(f"Legitimacy: {assessment.get('legitimacy_status', 'N/A')}")
            print(f"Employer Review: {assessment.get('employer_review', 'N/A')}")
            print(f"Growth: {assessment.get('growth', 'N/A')}")
            print(f"Revenue: {assessment.get('revenue', 'N/A')}")
            print(f"Scale: {assessment.get('scale', 'N/A')}")
            
            if job_role and assessment.get('job_role_assessment'):
                print(f"Job Role Assessment: {assessment.get('job_role_assessment', 'N/A')}")
            
            print(f"Red Flags: {assessment.get('red_flags', 'N/A')}")
            print(f"Positive Indicators: {assessment.get('positive_indicators', 'N/A')}")
        else:
            print(f"\nAssessment: {result.get('assessment', 'No assessment available')}")
        
        if 'output_file' in result:
            print(f"\n✅ Results saved to: {result['output_file']}")
    else:
        print("No company name provided.")
    
    print("\n" + "=" * 50)
    print("Check Complete")