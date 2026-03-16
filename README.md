# Satellite Intelligence Agent

An AI-powered satellite imagery analysis agent that answers plain English 
questions about vegetation, land cover, and environmental change anywhere 
on Earth using free Sentinel-2 data.

## What it does
- Searches the Microsoft Planetary Computer STAC catalog for Sentinel-2 imagery
- Streams only the pixels needed — no full scene downloads
- Calculates NDVI (vegetation health index) for any location
- Detects vegetation change between two time periods
- Generates colour-coded maps rendered directly in the browser
- Answers questions in plain English with statistical analysis

## Example questions
- "What is the vegetation health of Sheffield in summer 2024?"
- "Show me how vegetation changed in Manchester between summer and winter 2024"
- "Analyse the land cover of Edinburgh in June 2024"
- "Compare vegetation in Birmingham between 2023 and 2024"
- "How green is London in August 2024?"

## Tech stack
- Anthropic Claude API — natural language to satellite analysis
- pystac-client — search Microsoft Planetary Computer STAC catalog
- stackstac — stream COG pixels into Xarray data cubes
- NumPy — NDVI calculation
- Matplotlib — map generation
- Streamlit — web interface
- Sentinel-2 L2A — free ESA satellite imagery (10m resolution, 5-day revisit)

## Applications
- ESG supply chain deforestation monitoring
- Agricultural crop health assessment
- Urban green space analysis
- Climate change vegetation impact studies
- Environmental compliance reporting

## Setup
1. Clone the repository
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install anthropic pystac-client stackstac planetary-computer numpy matplotlib streamlit python-dotenv xarray rasterio`
5. Add `ANTHROPIC_API_KEY=your-key` to `.env`
6. Run: `streamlit run app.py`

## Built by
Ridwan Shittu - Geospatial consultant