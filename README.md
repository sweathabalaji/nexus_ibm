# 🚀 Resume Analyzer API

A comprehensive FastAPI-based resume analysis system that provides intelligent resume scoring, company verification, and detailed ATS compatibility analysis. This application integrates multiple AI-powered analysis tools to help job seekers optimize their resumes and verify potential employers.

## ✨ Features

### 🎯 Core Functionality
- **Resume-Job Match Analysis**: Score how well a resume matches a job description (0-100 scale)
- **Company Verification**: Verify legitimacy of companies mentioned in resumes
- **Detailed ATS Analysis**: Comprehensive ATS compatibility scoring and recommendations
- **Company Legitimacy Check**: In-depth analysis of company reputation and employment practices

### 🔍 Analysis Types
1. **Basic Analysis**: Resume + Job Description matching with company verification
2. **Detailed Resume Analysis**: ATS scoring, keyword analysis, action verb assessment
3. **Company Legitimacy Check**: Standalone company verification with detailed insights
4. **File Upload Support**: PDF and DOCX file processing

### 🎨 Multiple Interfaces
- **REST API**: FastAPI-based web service
- **Command Line**: Direct script execution with various modes
- **Batch Processing**: Support for multiple file formats

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- API Keys for:
  - Groq API (GROQ_API_KEY)
  - Tavily Search API (TAVILY_API_KEY)
  - Optional: MoonShot AI API (MOONSHOT_API_KEY)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sweathabalaji/nexus_ibm.git
cd nexus_ibm
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MOONSHOT_API_KEY=your_moonshot_api_key_here  # Optional
```

## 🚀 Usage

### FastAPI Server
Start the web server:
```bash
python app.py --api
```
Access the API documentation at: `http://localhost:8000/docs`

### Command Line Interface

#### Basic Resume Analysis
```bash
python app.py
```
Analyzes predefined resume and job description files.

#### Detailed Resume Analysis
```bash
python app.py --detailed-analysis /path/to/resume.pdf
```

#### Company Legitimacy Check
```bash
python app.py --company-check "Company Name" "Job Role"
```

#### Help
```bash
python app.py --help
```

## 📚 API Endpoints

### 🏠 General Endpoints

#### Health Check
```http
GET /health
```
Returns API health status and configuration.

#### Root Information
```http
GET /
```
Returns API information and available endpoints.

### 📊 Analysis Endpoints

#### Basic Resume Analysis
```http
POST /analyze
Content-Type: multipart/form-data

- resume_file: PDF/DOCX file
- job_description_file: PDF/DOCX file
```

#### Text-based Analysis
```http
POST /analyze-text
Content-Type: application/json

{
    "resume_text": "Resume content...",
    "job_description_text": "Job description content..."
}
```

#### Detailed Resume Analysis (Text)
```http
POST /detailed-resume-analysis
Content-Type: application/json

{
    "resume_text": "Resume content..."
}
```

#### Detailed Resume Analysis (File)
```http
POST /detailed-resume-analysis-file
Content-Type: multipart/form-data

- resume_file: PDF/DOCX file
```

#### Company Legitimacy Check
```http
POST /company-legitimacy-check
Content-Type: application/json

{
    "company_name": "Google",
    "job_role": "Software Engineer"  // Optional
}
```

## 📋 Response Formats

### Basic Analysis Response
```json
{
    "company_verification": [
        {
            "company_name": "Google",
            "legitimacy_status": "Legitimate",
            "payment_status": "Paid",
            "key_findings": "Well-established technology company with excellent employee benefits",
            "reputation_score": 9
        }
    ],
    "job_match_analysis": {
        "match_score": 85.5,
        "summary": "Strong technical background matches requirements. Missing some advanced cloud experience. Overall good fit for the role."
    }
}
```

### Detailed Resume Analysis Response
```json
{
    "resume_analysis": {
        "analysis_timestamp": "2025-01-08",
        "ats_score": 78,
        "ats_details": {
            "ats_score": 78,
            "ats_factors": {
                "formatting_score": 20,
                "keyword_density": 18,
                "section_organization": 22,
                "readability": 18
            },
            "ats_recommendations": [
                "Add more industry-specific keywords",
                "Improve section header consistency"
            ]
        },
        "keyword_analysis": {
            "keyword_analysis": {
                "technical_skills": ["Python", "JavaScript", "React"],
                "soft_skills": ["Leadership", "Communication"],
                "industry_keywords": ["Agile", "DevOps"],
                "missing_keywords": ["Cloud Computing", "Microservices"],
                "keyword_density_score": 72
            }
        },
        "action_verb_analysis": {
            "action_verb_analysis": {
                "strong_verbs": ["Implemented", "Optimized", "Led"],
                "weak_verbs": ["Responsible for", "Worked on"],
                "verb_strength_score": 75,
                "verb_variety_score": 68
            }
        },
        "strengths_weaknesses": {
            "strengths": [
                "Strong technical skill set",
                "Quantifiable achievements",
                "Relevant project experience"
            ],
            "weaknesses": [
                "Missing industry certifications",
                "Limited leadership experience descriptions"
            ],
            "overall_impression": "Solid technical candidate with room for improvement in presentation",
            "competitiveness_score": 76
        },
        "improvement_recommendations": {
            "priority_improvements": [
                {
                    "category": "Content",
                    "issue": "Missing quantified achievements",
                    "recommendation": "Add specific metrics to demonstrate impact",
                    "priority": "High",
                    "example": "Increased system performance by 40%"
                }
            ]
        },
        "overall_grade": "B (Good)"
    }
}
```

### Company Legitimacy Response
```json
{
    "company_name": "Google",
    "job_role": "Software Engineer",
    "assessment": {
        "one_line_summary": "Global technology company specializing in internet services and products",
        "legitimacy_status": "Legitimate",
        "employer_review": "Excellent employer with 4.4/5 rating, known for innovation and employee benefits",
        "growth": "Consistent growth with expanding market presence",
        "revenue": "Over $280 billion annual revenue",
        "scale": "Over 150,000 employees worldwide",
        "job_role_assessment": "High demand for software engineers with competitive compensation",
        "red_flags": "None identified",
        "positive_indicators": "Strong market position, excellent employee reviews, competitive benefits"
    },
    "status": "success"
}
```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for AI analysis
- `TAVILY_API_KEY`: Required for web search functionality
- `MOONSHOT_API_KEY`: Optional, for advanced resume analysis

### File Paths (for CLI mode)
Edit these variables in `app.py` for default file processing:
```python
RESUME_FILE_PATH = "/path/to/your/resume.pdf"
JOB_DESCRIPTION_FILEPATH = "/path/to/job/description.pdf"
```

## 📁 Project Structure

```
.
├── app.py                 # Main FastAPI application
├── RA.py                 # Original resume analyzer (integrated)
├── job.py                # Original job checker (integrated)
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .env                 # Environment variables (create this)
├── render.yaml          # Deployment configuration
└── uploads/             # File upload directory
```

## 🔍 Analysis Features Explained

### 1. ATS Score Breakdown
- **Formatting Score (0-25)**: Resume layout and structure
- **Keyword Density (0-25)**: Relevant keyword usage
- **Section Organization (0-25)**: Proper section headers and flow
- **Readability (0-25)**: Text clarity and scanning ease

### 2. Company Verification Factors
- **Legitimacy Status**: Legitimate/Suspicious/Unknown
- **Payment Status**: Paid/Unpaid/Student Paid Fee/Unknown
- **Reputation Score**: 1-10 scale based on employee reviews
- **Growth Assessment**: Company trajectory and stability

### 3. Job Match Scoring
- Skills alignment with requirements
- Experience level compatibility
- Educational background relevance
- Project and achievement relevance
- Gap identification

## 🚨 Error Handling

The API includes comprehensive error handling:
- File format validation
- Text length validation
- API key verification
- Graceful failure recovery
- Detailed error messages

## 📊 Output Files

The application generates JSON reports:
- `complete_analysis_report.json`: Full analysis results
- `detailed_resume_analysis.json`: Detailed ATS analysis
- `{company_name}_legitimacy_check.json`: Company verification results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the [Issues](https://github.com/sweathabalaji/nexus_ibm/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] Support for more file formats (TXT, RTF)
- [ ] Batch processing capabilities
- [ ] Resume template suggestions
- [ ] Industry-specific analysis
- [ ] Real-time collaboration features
- [ ] Integration with job boards
- [ ] Advanced analytics dashboard

## 🙏 Acknowledgments

- OpenAI for language models
- Groq for fast inference
- Tavily for search capabilities
- FastAPI for the web framework
- LangChain for AI orchestration

---

**Made with ❤️ for better career outcomes**
