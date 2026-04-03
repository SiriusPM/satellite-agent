# Satellite Intelligence Agent

An AI-powered satellite imagery analysis agent that produces professional 
ESG environmental reports using free Sentinel-2 data from two cloud platforms.

## What it does

- Accepts plain English questions about any location on Earth
- Autonomously decides which combination of tools to use based on the question
- Searches petabytes of Sentinel-2 satellite imagery via STAC catalog
- Streams only the pixels needed — no full scene downloads required
- Runs multi-year time series analysis entirely on Google Earth Engine servers
- Calculates five ESG satellite indices simultaneously per location
- Detects vegetation change between any two time periods
- Generates year-by-year ESG performance tables with AI-written interpretation
- Assigns ESG environmental grades with written justification
- Produces professional multi-panel satellite dashboard maps
- Displays all maps and reports directly in a web chat interface
- Maintains conversation memory for follow-up questions
- Cites exact data sources, dates, and cloud cover for every analysis
- Calls multiple tools in parallel when a question requires it
- Handles the full ESG monitoring workflow end to end — from raw imagery to client report

## Case study — Birmingham ESG Environmental Assessment (2020–2024)

**Question asked:** "Run a full ESG assessment of Birmingham from 2020 to 2024"

**What the agent did autonomously:**
- Searched for cloud-free Sentinel-2 imagery across 5 years
- Called Google Earth Engine to calculate 37 monthly NDVI data points
- Called Microsoft Planetary Computer to run annual ESG snapshots for each year
- Synthesised all results into a structured report — no human intervention

**What it found:**

| Year | NDVI | Key finding |
|---|---|---|
| 2020 | 0.341 | COVID-19 baseline — reduced urban activity |
| 2021 | 0.425 | Peak vegetation — lockdown recovery effect |
| 2022 | 0.203 | Sharp drop — UK record drought detected |
| 2023 | 0.250 | Recovery phase begins |
| 2024 | 0.261 | Continued recovery, improving trend |

**ESG grade assigned: B+**

The agent correctly identified the 2022 UK drought from satellite data alone —
without being told about it. NDVI dropped from 0.425 to 0.203, a 52% reduction,
consistent with the hottest year on record in the UK. By 2024 vegetation had
partially recovered, earning Birmingham a B+ for environmental resilience.

**Time to deliver:** ~3 minutes from question to full report  
**Data cost:** £0 — all imagery is free via ESA Copernicus Programme  
**Traditional equivalent:** 2–3 days of manual GIS analysis

## The four tools Claude orchestrates autonomously

| Tool | Platform | Use case |
|---|---|---|
| `search_satellite_imagery` | STAC | Find available scenes by location and date |
| `calculate_ndvi` | STAC | Single scene vegetation map, change detection |
| `calculate_indices` | STAC | Full ESG dashboard — NDVI, NDWI, NDBI, NBR, EVI |
| `get_ndvi_timeseries` | Google Earth Engine | Multi-year monthly NDVI trend analysis |

## The five ESG indices

| Index | Measures | ESG application |
|---|---|---|
| NDVI | Vegetation health | Deforestation, biodiversity baseline |
| NDWI | Surface water | Flood risk, water dependency (TNFD) |
| NDBI | Urban built-up areas | Land use change, urban expansion |
| NBR | Burn and fire damage | Wildfire risk, post-fire assessment |
| EVI | Enhanced vegetation | Carbon stock, dense forest monitoring |

## Example questions

- "Run a full ESG assessment of Birmingham from 2020 to 2024"
- "What is the vegetation health of Sheffield in summer 2024?"
- "Show me the 5-year NDVI trend for Manchester from 2020 to 2024"
- "How has vegetation changed in Leeds since 2019?"
- "Show me how vegetation changed in Sheffield between summer and winter 2024"
- "Run a complete satellite index dashboard for Cardiff in July 2024"

## Example output

The agent autonomously:
1. Searches for cloud-free imagery across multiple years
2. Runs GEE time series for long-term trend data
3. Runs STAC analysis for detailed annual snapshots
4. Synthesises findings into a structured ESG report
5. Assigns an ESG grade with written justification
6. Displays satellite maps inline in the chat

## Tech stack

- Anthropic Claude API — natural language orchestration and ESG report writing
- pystac-client + stackstac — stream COG pixels from Microsoft Planetary Computer
- Google Earth Engine Python API — multi-year time series on Google's cloud
- NumPy — index calculations
- Matplotlib — satellite map generation
- Streamlit — web chat interface
- Sentinel-2 L2A — free ESA imagery, 10m resolution, 5-day revisit

## ESG applications

- TCFD physical climate risk assessment
- TNFD nature dependency and impact mapping
- Supply chain deforestation monitoring
- Agricultural crop health assessment
- Urban green infrastructure analysis
- Post-wildfire damage assessment
- Annual environmental compliance reporting

## Setup

1. Clone the repository
2. Create virtual environment: `python3.13 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install anthropic pystac-client stackstac planetary-computer earthengine-api geemap numpy matplotlib streamlit python-dotenv xarray rasterio`
5. Authenticate GEE: `earthengine authenticate --auth_mode notebook`
6. Add `ANTHROPIC_API_KEY=your-key` to `.env`
7. Update GEE project ID in `app.py` line 16
8. Run: `streamlit run app.py`

## Built by

Ridwan Shittu — Geospatial consultant combining satellite remote sensing 
expertise with modern AI to build environmental monitoring tools for ESG 
reporting and commercial location intelligence.