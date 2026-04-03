import streamlit as st
import anthropic
import pystac_client
import stackstac
import planetary_computer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import ee
from dotenv import load_dotenv

load_dotenv()

ee.Initialize(project='delta-blue-carbon')

st.set_page_config(
    page_title="Satellite Intelligence Agent",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Intelligence Agent")
st.caption("Ask any question about vegetation, land cover, or environmental change anywhere on Earth.")

@st.cache_resource
def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

catalog = get_catalog()

SYSTEM_PROMPT = """You are a satellite imagery analyst with access to 
Sentinel-2 satellite data via two platforms:
1. Microsoft Planetary Computer (STAC) — for single scene analysis
2. Google Earth Engine (GEE) — for multi-year time series analysis

You have four tools:
1. search_satellite_imagery — find available scenes for a location and date
2. calculate_ndvi — single scene vegetation analysis, optionally compare two dates
3. calculate_indices — full ESG dashboard (NDVI, NDWI, NDBI, NBR, EVI)
4. get_ndvi_timeseries — multi-year monthly NDVI trend via Google Earth Engine

When to use each tool:
- Single date vegetation question → calculate_ndvi
- Full ESG site assessment → calculate_indices
- Change detection between two dates → calculate_ndvi with date1 and date2
- Multi-year trend, historical analysis, long time series → get_ndvi_timeseries
- Fire, flood, urban expansion → calculate_indices

The get_ndvi_timeseries tool is powerful for:
- "How has vegetation changed over the last 5 years?"
- "Show me the long-term NDVI trend"
- "Has deforestation occurred since 2018?"
- "What is the seasonal pattern of vegetation?"

Common UK city bounding boxes — use these exactly:
- Sheffield: [-1.55, 53.35, -1.35, 53.45]
- London: [-0.15, 51.45, 0.05, 51.55]
- Manchester: [-2.25, 53.44, -2.18, 53.52]
- Birmingham: [-1.95, 52.44, -1.82, 52.52]
- Leeds: [-1.58, 53.78, -1.48, 53.85]
- Edinburgh: [-3.25, 55.92, -3.15, 55.98]
- Cardiff: [-3.22, 51.46, -3.14, 51.52]

Always use these exact small bounding boxes — do not make them larger.
Always explain what index values mean in plain English.
Always mention the date of imagery and data source used."""

tools = [
    {
        "name": "search_satellite_imagery",
        "description": """Search for available Sentinel-2 satellite imagery 
        over a location and time period. Always call this first before 
        single-scene analysis.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box [min_lon, min_lat, max_lon, max_lat]"
                },
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                "max_cloud_cover": {"type": "number", "default": 30}
            },
            "required": ["bbox", "start_date", "end_date"]
        }
    },
    {
        "name": "calculate_ndvi",
        "description": """Calculate NDVI vegetation index from a single Sentinel-2 
        scene and generate a map. Optionally compare two dates for change detection.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box [min_lon, min_lat, max_lon, max_lat]"
                },
                "date1": {"type": "string", "description": "Primary date YYYY-MM-DD"},
                "date2": {"type": "string", "description": "Second date for change detection"},
                "location_name": {"type": "string"}
            },
            "required": ["bbox", "date1", "location_name"]
        }
    },
    {
        "name": "calculate_indices",
        "description": """Calculate all five ESG satellite indices:
        NDVI (vegetation), NDWI (water), NDBI (urban), NBR (fire), EVI (enhanced vegetation).
        Produces a multi-panel ESG dashboard map.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box [min_lon, min_lat, max_lon, max_lat]"
                },
                "date": {"type": "string", "description": "Date YYYY-MM-DD"},
                "location_name": {"type": "string"}
            },
            "required": ["bbox", "date", "location_name"]
        }
    },
    {
        "name": "get_ndvi_timeseries",
        "description": """Calculate monthly NDVI time series over multiple years 
        using Google Earth Engine. Use for long-term trend analysis, 
        deforestation monitoring, and historical vegetation change.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box [min_lon, min_lat, max_lon, max_lat]"
                },
                "start_year": {"type": "integer", "description": "Start year e.g. 2019"},
                "end_year": {"type": "integer", "description": "End year e.g. 2024"},
                "location_name": {"type": "string"}
            },
            "required": ["bbox", "start_year", "end_year", "location_name"]
        }
    }
]


def search_satellite_imagery(bbox, start_date, end_date, max_cloud_cover=30):
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}}
        )
        items = list(search.items())
        if not items:
            return json.dumps({"error": "No scenes found"})
        results = [{
            "id": item.id,
            "date": item.datetime.strftime("%Y-%m-%d"),
            "cloud_cover": round(item.properties['eo:cloud_cover'], 1)
        } for item in items[:10]]
        return json.dumps({
            "total_scenes": len(items),
            "scenes": results,
            "best_scene": min(results, key=lambda x: x['cloud_cover'])
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_scene_bands(bbox, date, bands):
    import datetime
    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
    start = (date_obj - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    end = (date_obj + datetime.timedelta(days=10)).strftime("%Y-%m-%d")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 30}}
    )
    items = list(search.items())
    if not items:
        return None, None, None

    # Use only the single best item to avoid CRS conflicts between tiles
    best = min(items, key=lambda x: x.properties['eo:cloud_cover'])

    stack = stackstac.stack(
        [best],
        assets=bands,
        bounds_latlon=bbox,
        resolution=60,
        fill_value=np.nan,
        epsg=32630  # Force UTM Zone 30N — consistent CRS for UK
    )

    scene = stack.isel(time=0).compute()
    actual_date = best.datetime.strftime("%Y-%m-%d")
    cloud = round(best.properties['eo:cloud_cover'], 1)
    return scene, actual_date, cloud


def crop_valid(arr):
    valid_rows = ~np.all(np.isnan(arr), axis=1)
    valid_cols = ~np.all(np.isnan(arr), axis=0)
    if not np.any(valid_rows) or not np.any(valid_cols):
        return arr
    return arr[valid_rows][:, valid_cols]


def get_ndvi_for_date(bbox, date):
    scene, actual_date, cloud = get_scene_bands(bbox, date, ["B04", "B08"])
    if scene is None:
        return None, None, None
    red = scene.sel(band="B04").values.astype(float) / 10000
    nir = scene.sel(band="B08").values.astype(float) / 10000
    ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))
    ndvi = crop_valid(ndvi)
    return ndvi, actual_date, cloud


def calculate_ndvi(bbox, date1, location_name, date2=None):
    try:
        ndvi1, actual_date1, cloud1 = get_ndvi_for_date(bbox, date1)
        if ndvi1 is None:
            return json.dumps({"error": f"No imagery found near {date1}"})
        valid = ~np.isnan(ndvi1)
        if np.sum(valid) == 0:
            return json.dumps({"error": "No valid pixels found"})
        stats1 = {
            "date": actual_date1,
            "cloud_cover": cloud1,
            "mean_ndvi": round(float(np.nanmean(ndvi1)), 3),
            "urban_pct": round(float(np.sum((ndvi1 >= 0) & (ndvi1 < 0.2)) / np.sum(valid) * 100), 1),
            "sparse_veg_pct": round(float(np.sum((ndvi1 >= 0.2) & (ndvi1 < 0.5)) / np.sum(valid) * 100), 1),
            "dense_veg_pct": round(float(np.sum(ndvi1 >= 0.5) / np.sum(valid) * 100), 1)
        }
        safe_name = location_name.lower().replace(' ', '_')
        if date2:
            ndvi2, actual_date2, cloud2 = get_ndvi_for_date(bbox, date2)
            if ndvi2 is None:
                return json.dumps({"error": f"No imagery found near {date2}"})
            min_rows = min(ndvi1.shape[0], ndvi2.shape[0])
            min_cols = min(ndvi1.shape[1], ndvi2.shape[1])
            ndvi1 = ndvi1[:min_rows, :min_cols]
            ndvi2 = ndvi2[:min_rows, :min_cols]
            ndvi_diff = ndvi2 - ndvi1
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(ndvi1, cmap='RdYlGn', vmin=-0.2, vmax=0.8, aspect='auto')
            axes[0].set_title(f'NDVI — {actual_date1}\nCloud: {cloud1}%')
            axes[0].set_axis_off()
            axes[1].imshow(ndvi2, cmap='RdYlGn', vmin=-0.2, vmax=0.8, aspect='auto')
            axes[1].set_title(f'NDVI — {actual_date2}\nCloud: {cloud2}%')
            axes[1].set_axis_off()
            img = axes[2].imshow(ndvi_diff, cmap='RdBu', vmin=-0.4, vmax=0.4, aspect='auto')
            axes[2].set_title(f'Vegetation Change\n{actual_date1} → {actual_date2}')
            axes[2].set_axis_off()
            plt.colorbar(img, ax=axes[2], label='NDVI Change', shrink=0.8)
            plt.suptitle(f'{location_name} — Vegetation Change Analysis', fontsize=13)
            plt.tight_layout()
            map_path = f"{safe_name}_change.png"
            plt.savefig(map_path, dpi=150, bbox_inches='tight')
            plt.close()
            return json.dumps({
                "period1": stats1,
                "period2": {"date": actual_date2, "cloud_cover": cloud2,
                            "mean_ndvi": round(float(np.nanmean(ndvi2)), 3)},
                "mean_change": round(float(np.nanmean(ndvi_diff)), 3),
                "map_saved": map_path
            })
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            img = ax.imshow(ndvi1, cmap='RdYlGn', vmin=-0.2, vmax=0.8, aspect='auto')
            plt.colorbar(img, ax=ax, label='NDVI', shrink=0.7)
            ax.set_title(f'{location_name} — Vegetation Health\n{actual_date1} (Cloud: {cloud1}%)', fontsize=13)
            ax.set_axis_off()
            plt.tight_layout()
            map_path = f"{safe_name}_ndvi.png"
            fig.savefig(map_path, dpi=150, bbox_inches='tight')
            plt.close()
            return json.dumps({"stats": stats1, "map_saved": map_path})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate_indices(bbox, date, location_name):
    try:
        scene, actual_date, cloud = get_scene_bands(
            bbox, date, ["B02", "B03", "B04", "B08", "B11", "B12"]
        )
        if scene is None:
            return json.dumps({"error": f"No imagery found near {date}"})

        def get_band(b):
            arr = scene.sel(band=b).values.astype(float) / 10000
            return crop_valid(arr)

        blue  = get_band("B02")
        green = get_band("B03")
        red   = get_band("B04")
        nir   = get_band("B08")
        swir  = get_band("B11")
        swir2 = get_band("B12")

        def safe_index(a, b):
            return np.where((a + b) == 0, np.nan, (a - b) / (a + b))

        ndvi = safe_index(nir, red)
        ndwi = safe_index(green, nir)
        ndbi = safe_index(swir, nir)
        nbr  = safe_index(nir, swir2)
        evi  = np.where(
            (nir + 6*red - 7.5*blue + 1) == 0, np.nan,
            2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        )

        def stats(arr):
            valid = ~np.isnan(arr)
            if np.sum(valid) == 0:
                return {"mean": None}
            return {"mean": round(float(np.nanmean(arr)), 3)}

        indices_stats = {
            "date": actual_date,
            "cloud_cover": cloud,
            "ndvi": stats(ndvi),
            "ndwi": stats(ndwi),
            "ndbi": stats(ndbi),
            "nbr":  stats(nbr),
            "evi":  stats(evi)
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        panels = [
            (ndvi, 'RdYlGn', -0.2, 0.8, 'NDVI — Vegetation Health'),
            (ndwi, 'Blues',  -0.3, 0.5, 'NDWI — Surface Water'),
            (ndbi, 'RdBu_r', -0.5, 0.5, 'NDBI — Urban / Built-up'),
            (nbr,  'RdYlGn', -0.5, 0.8, 'NBR — Burn / Fire Detection'),
            (evi,  'YlGn',   -0.2, 0.8, 'EVI — Enhanced Vegetation'),
        ]
        for i, (arr, cmap, vmin, vmax, title) in enumerate(panels):
            ax = axes[i]
            img = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            plt.colorbar(img, ax=ax, shrink=0.7)
            ax.set_title(title, fontsize=11)
            ax.set_axis_off()

        axes[5].axis('off')
        axes[5].text(0.5, 0.7, location_name, ha='center', va='center',
                     fontsize=16, fontweight='bold', transform=axes[5].transAxes)
        axes[5].text(0.5, 0.5, actual_date, ha='center', va='center',
                     fontsize=13, transform=axes[5].transAxes)
        axes[5].text(0.5, 0.3, f'Cloud cover: {cloud}%', ha='center', va='center',
                     fontsize=11, color='gray', transform=axes[5].transAxes)

        plt.suptitle(f'{location_name} — ESG Satellite Index Dashboard\n{actual_date}',
                     fontsize=14, y=1.01)
        plt.tight_layout()
        safe_name = location_name.lower().replace(' ', '_')
        map_path = f"{safe_name}_esg_dashboard.png"
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close()
        return json.dumps({"indices": indices_stats, "map_saved": map_path})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_ndvi_timeseries(bbox, start_year, end_year, location_name):
    try:
        aoi = ee.Geometry.Rectangle(bbox)
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(f'{start_year}-01-01', f'{end_year}-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        )

        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        with_ndvi = collection.map(add_ndvi)

        months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                months.append(f'{year}-{month:02d}')

        dates = []
        ndvi_values = []

        for month_str in months:
            year, month = month_str.split('-')
            start = ee.Date(f'{year}-{month}-01')
            end = start.advance(1, 'month')
            monthly = with_ndvi.filterDate(start, end)
            count = monthly.size().getInfo()
            if count == 0:
                continue
            mean_ndvi = monthly.select('NDVI').mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=100,
                maxPixels=1e9
            ).getInfo()
            ndvi_val = mean_ndvi.get('NDVI')
            if ndvi_val is not None:
                dates.append(month_str)
                ndvi_values.append(ndvi_val)

        if not dates:
            return json.dumps({"error": "No valid NDVI data found"})

        stats = {
            "data_points": len(dates),
            "date_range": f"{dates[0]} to {dates[-1]}",
            "mean_ndvi": round(float(np.mean(ndvi_values)), 3),
            "min_ndvi": round(float(np.min(ndvi_values)), 3),
            "max_ndvi": round(float(np.max(ndvi_values)), 3),
            "trend": "increasing" if ndvi_values[-1] > ndvi_values[0] else "decreasing"
        }

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(range(len(dates)), ndvi_values,
                color='green', linewidth=1.5, alpha=0.8)
        ax.fill_between(range(len(dates)), ndvi_values,
                        alpha=0.2, color='green')

        if len(ndvi_values) >= 3:
            window = 3
            rolling = np.convolve(ndvi_values,
                                  np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(dates)),
                    rolling, color='darkgreen',
                    linewidth=2.5, label='3-month average')

        tick_positions = list(range(0, len(dates), 6))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([dates[i] for i in tick_positions],
                            rotation=45, ha='right')
        ax.set_ylabel('NDVI')
        ax.set_title(
            f'{location_name} — Monthly NDVI Time Series\n'
            f'Sentinel-2 via Google Earth Engine ({start_year}–{end_year})',
            fontsize=13
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.8)
        plt.tight_layout()

        safe_name = location_name.lower().replace(' ', '_')
        map_path = f"{safe_name}_timeseries.png"
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close()

        return json.dumps({"stats": stats, "map_saved": map_path})

    except Exception as e:
        return json.dumps({"error": str(e)})


def ask_satellite_agent(question, history):
    client = anthropic.Anthropic()
    history.append({"role": "user", "content": question})
    map_path = None
    tool_calls = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=history
        )

        if response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []

            for tool_use in tool_use_blocks:
                tool_calls.append(f"{tool_use.name}({json.dumps(tool_use.input)})")

                if tool_use.name == "search_satellite_imagery":
                    result = search_satellite_imagery(**tool_use.input)
                elif tool_use.name == "calculate_ndvi":
                    result = calculate_ndvi(**tool_use.input)
                elif tool_use.name == "calculate_indices":
                    result = calculate_indices(**tool_use.input)
                elif tool_use.name == "get_ndvi_timeseries":
                    result = get_ndvi_timeseries(**tool_use.input)
                else:
                    result = json.dumps({"error": f"Unknown tool: {tool_use.name}"})

                result_data = json.loads(result)
                if "map_saved" in result_data:
                    map_path = result_data["map_saved"]

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })

            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            final = next(b for b in response.content if hasattr(b, "text"))
            history.append({"role": "assistant", "content": final.text})
            return final.text, map_path, tool_calls


if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Example questions")
    examples = [
        "What is the vegetation health of Sheffield in summer 2024?",
        "Run a full ESG assessment of Birmingham in August 2024",
        "Show me the 5-year NDVI trend for Manchester from 2020 to 2024",
        "How has vegetation changed in Leeds since 2019?",
        "Show me how vegetation changed in Sheffield between summer and winter 2024",
        "Run a complete satellite index dashboard for Cardiff in July 2024"
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    st.divider()
    st.caption("STAC: single scene analysis")
    st.caption("GEE: multi-year time series")
    st.caption("Data: ESA Copernicus (free & open)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("map_path") and os.path.exists(msg["map_path"]):
            st.image(msg["map_path"], use_container_width=True)
        if msg.get("tool_calls"):
            with st.expander("Tools Claude used"):
                for call in msg["tool_calls"]:
                    st.code(call, language="python")

question = st.chat_input("Ask about vegetation, land cover, or environmental change anywhere on Earth...")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Analysing satellite imagery — this may take 1-2 minutes for time series..."):
            answer, map_path, tool_calls = ask_satellite_agent(
                question,
                st.session_state.history
            )

        st.markdown(answer)

        if map_path and os.path.exists(map_path):
            st.image(map_path, use_container_width=True)

        if tool_calls:
            with st.expander("Tools Claude used"):
                for call in tool_calls:
                    st.code(call, language="python")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "map_path": map_path,
        "tool_calls": tool_calls
    })