# Satellite Intelligence Agent

An AI-powered satellite imagery analysis agent that answers plain English 
questions about vegetation, water, urban expansion, fire damage, and 
environmental change anywhere on Earth using free Sentinel-2 data.

## What it does
- Searches Microsoft Planetary Computer STAC catalog for Sentinel-2 imagery
- Streams only the pixels needed — no full scene downloads
- Calculates five ESG satellite indices for any location
- Detects vegetation change between two time periods
- Generates professional multi-panel ESG dashboard maps
- Answers questions in plain English with statistical analysis

## The five ESG indices

| Index | What it measures | ESG use case |
|---|---|---|
| NDVI | Vegetation health and density | Deforestation, biodiversity baseline |
| NDWI | Surface water presence | Flood risk, water dependency (TNFD) |
| NDBI | Urban and built-up areas | Land use change, urban expansion |
| NBR | Burn and fire damage | Wildfire risk, post-fire assessment |
| EVI | Enhanced vegetation (dense canopy) | Carbon stock, tropical forest |

## Example questions
- "What is the vegetation health of Sheffield in summer 2024?"
- "Run a full ESG assessment of Birmingham in August 2024"
- "Show me how vegetation changed in Manchester between summer and winter 2024"
- "Is there any surface water visible near Edinburgh in June 2024?"
- "Show me the urban expansion in London in summer 2024"
- "Run a complete satellite index dashboard for Cardiff in July 2024"

## Tech stack
- Anthropic Claude API — natural language to satellite analysis
- pystac-client — search Microsoft Planetary Computer STAC catalog
- stackstac — stream COG pixels into Xarray data cubes
- NumPy — index calculations
- Matplotlib — dashboard map generation
- Streamlit — web interface
- Sentinel-2 L2A — free ESA satellite imagery (10m resolution, 5-day revisit)

## Applications
- ESG supply chain deforestation monitoring
- TCFD physical climate risk assessment
- TNFD nature dependency mapping
- Agricultural crop health assessment
- Urban green space and impervious surface analysis
- Post-wildfire damage assessment for insurance clients
- Environmental compliance reporting

## Setup
1. Clone the repository
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies:
   `pip install anthropic pystac-client stackstac planetary-computer numpy matplotlib streamlit python-dotenv xarray rasterio`
5. Add `ANTHROPIC_API_KEY=your-key` to `.env`
6. Run: `streamlit run app.py`

## Built by
Ridwan Shittu - Geospatial consultant combining satellite remote sensing expertise 
with modern AI to build environmental monitoring tools for ESG reporting 
and commercial location intelligence.