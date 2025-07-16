# Medical Prescription Analyzer

## Overview
This application is an AI-powered medical prescription analyzer that uses Google Gemini and RAG (Retrieval-Augmented Generation) technology to provide comprehensive analysis of medication prescriptions. The system evaluates multiple risk factors to help healthcare professionals make informed decisions about patient safety.

## Key Features

### 1. Comprehensive Risk Analysis
The application analyzes prescriptions across multiple dimensions:
- **Drug Interactions**: Detection and classification of interactions (Major/Moderate/Minor)
- **Dosage Issues**: Identification of overdosage and underdosage
- **Contraindications**: Detection of absolute and relative contraindications
- **Therapeutic Redundancy**: Identification of duplicate therapies and unnecessary combinations
- **Administration Routes**: Analysis of inappropriate administration routes and risks
- **Treatment Duration**: Evaluation of excessive or insufficient treatment periods
- **Potential Side Effects**: Analysis of cumulative side effects and severe risks
- **Monitoring Recommendations**: Recommendations for clinical and laboratory monitoring

### 2. Advanced Visualizations
- Interactive tables with filtering capabilities
- Distribution charts for each risk category
- Timeline visualizations for administration routes
- Heat maps for side effect analysis
- Global risk scoring dashboard

### 3. Technical Features
- Automatic medication extraction from free-text prescriptions
- Integration with medical guidelines and references via vector search
- Detailed reporting with medical sources
- Structured JSON output for each risk category
- Caching system for performance optimization

## Architecture
- **Frontend**: Streamlit web interface with interactive components
- **AI Engine**: Google Gemini for LLM analysis
- **RAG System**: FAISS for vector-based retrieval of medical guidelines
- **Modular Structure**: 
  - `config/`: Configuration settings
  - `core/`: Core system functions
  - `ai/`: AI analysis modules
  - `data/`: Data processing and RAG
  - `ui/`: User interface components
  - `utils/`: Utility functions

## Medical Reference Sources
- STOPP/START criteria
- Beers criteria
- Laroche list
- PRISCUS list
- ONC DDI guidelines

## Installation and Usage

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Setup
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your API key in `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Application
```
streamlit run main.py
```

## Analysis Process
1. Input your prescription text in the analysis page
2. The system automatically extracts medications
3. The AI engine analyzes multiple risk factors
4. Results are displayed with interactive visualizations
5. Download detailed reports for clinical use

## Development
This application is structured in a modular way to enable easy extension of risk analysis categories. Each analyzer follows the same pattern:
1. Analysis module in `ai/` directory
2. UI components in `ui/components/`
3. Prompt template in `utils/constants.py`
4. Integration in main analysis workflow

## License
This project is for educational and research purposes.

## Contributors
- Medical Prescription Analysis Team
- ISIS Engineering School
