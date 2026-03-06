import streamlit as st
import numpy as np
import heapq
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
from matplotlib import colors

# --- Configuration ---
GRID_ROWS = 25
GRID_COLS = 35
MODEL_NAME = 'gemini-2.5-flash' # Using a stable model name available in the environment

# --- Pathfinding Logic (A*) ---
def a_star(grid, start, end):
    rows, cols = grid.shape
    start_node = tuple(start)
    end_node = tuple(end)
    
    if grid[start_node] == 1 or grid[end_node] == 1:
        return None # Start or End is a wall

    # Priority Queue: (f_score, g_score, r, c)
    open_set = []
    heapq.heappush(open_set, (0, 0, start_node[0], start_node[1]))
    
    came_from = {}
    g_score = {start_node: 0}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while open_set:
        _, current_g, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            return path
            
        if current_g > g_score.get(current, float('inf')):
            continue
            
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 1: # Wall
                    continue
                
                neighbor = (nr, nc)
                tentative_g = current_g + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = abs(nr - end_node[0]) + abs(nc - end_node[1])
                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))
                    
    return None

# --- Gemini Logic ---
def generate_floor_plan(description):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API Key not found in environment variables.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""
    Generate a 2D grid layout for a floor map with dimensions {GRID_ROWS}x{GRID_COLS}.
    The grid should represent:
    0 = Empty walkable space
    1 = Wall / Obstacle

    Context: {description}
    
    Ensure the walls form logical structures (rooms, hallways). 
    Do not place walls on every single edge, make it navigable.
    Return strictly a JSON object with a 'layout' property containing the 2D array.
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        import json
        text = response.text
        data = json.loads(text)
        layout = data.get('layout')
        
        if layout and len(layout) == GRID_ROWS and len(layout[0]) == GRID_COLS:
            return np.array(layout)
        
        # Fallback resizing if dimensions don't match
        if layout:
            new_layout = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
            r_limit = min(len(layout), GRID_ROWS)
            c_limit = min(len(layout[0]), GRID_COLS)
            for r in range(r_limit):
                for c in range(c_limit):
                    new_layout[r, c] = layout[r][c]
            return new_layout
            
        return None
    except Exception as e:
        st.error(f"Error generating map: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="SmartFloor PathFinder", layout="wide")

st.title("SmartFloor PathFinder 🗺️")
st.markdown("Generate floor plans with AI and find the shortest path using A*.")

# Initialize Session State
if 'grid_version' not in st.session_state:
    st.session_state.grid_version = 0

if 'grid' not in st.session_state:
    # Default grid with some walls
    initial_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    # Simple default walls
    initial_grid[7, :5] = 1
    initial_grid[7, 6:17] = 1 # Gap at 5
    initial_grid[7, 18:29] = 1 # Gap at 17 (center approx)
    initial_grid[7, 30:] = 1
    
    initial_grid[:, 10] = 1
    initial_grid[12, 10] = 0 # Door
    
    st.session_state.grid = initial_grid

if 'path' not in st.session_state:
    st.session_state.path = None

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    
    st.subheader("Start / End Positions")
    col1, col2 = st.columns(2)
    with col1:
        start_row = st.number_input("Start Row", 0, GRID_ROWS-1, 3)
        start_col = st.number_input("Start Col", 0, GRID_COLS-1, 3)
    with col2:
        end_row = st.number_input("End Row", 0, GRID_ROWS-1, GRID_ROWS-4)
        end_col = st.number_input("End Col", 0, GRID_COLS-1, GRID_COLS-4)
        
    start_pos = (start_row, start_col)
    end_pos = (end_row, end_col)
    
    st.divider()
    
    st.subheader("AI Generation")
    prompt = st.text_area("Description", "An intricate office floor plan with a few meeting rooms and corridors.")
    if st.button("Generate Floor Plan"):
        with st.spinner("Generating map with Gemini..."):
            new_layout = generate_floor_plan(prompt)
            if new_layout is not None:
                st.session_state.grid = new_layout
                st.session_state.path = None
                st.session_state.grid_version += 1 # Force editor reset
                st.success("Map generated!")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Find Path", type="primary"):
            st.session_state.path = a_star(st.session_state.grid, start_pos, end_pos)
            if st.session_state.path is None:
                st.error("No path found!")
    with col4:
        if st.button("Clear Path"):
            st.session_state.path = None
            
    if st.button("Clear All Walls"):
        st.session_state.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        st.session_state.path = None
        st.session_state.grid_version += 1 # Force editor reset

# Main Area
col_main, col_legend = st.columns([4, 1])

with col_main:
    # Visualization
    # Create a custom colormap: 0=White (Empty), 1=Black (Wall), 2=Green (Start), 3=Red (End), 4=Blue (Path)
    
    # We create a visualization grid separate from the logical grid
    viz_grid = st.session_state.grid.copy() * 10 # Scale walls to 10 for distinct color
    
    # Overlay Path
    if st.session_state.path:
        for r, c in st.session_state.path:
            viz_grid[r, c] = 5 # Path value
            
    # Overlay Start/End
    viz_grid[start_pos] = 20 # Start
    viz_grid[end_pos] = 30 # End
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors
    # 0: White (Empty)
    # 5: Blue (Path)
    # 10: Black (Wall)
    # 20: Green (Start)
    # 30: Red (End)
    
    cmap = colors.ListedColormap(['white', '#3b82f6', '#1e293b', '#22c55e', '#ef4444'])
    bounds = [-1, 2, 7, 15, 25, 35]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(viz_grid, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='#e2e8f0', linewidth=1)
    ax.set_xticks(np.arange(-.5, GRID_COLS, 1));
    ax.set_yticks(np.arange(-.5, GRID_ROWS, 1));
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    
    st.pyplot(fig)

with col_legend:
    st.markdown("### Legend")
    st.markdown("⬜ Empty Space")
    st.markdown("⬛ Wall")
    st.markdown("🟦 Path")
    st.markdown("🟩 Start")
    st.markdown("🟥 End")
    
    st.markdown("### Edit Grid")
    st.markdown("Modify the grid below (0=Empty, 1=Wall) to add/remove obstacles manually.")
    
    # Data Editor for manual adjustments
    # We transpose for better layout if needed, but let's keep it simple
    edited_grid = st.data_editor(
        st.session_state.grid,
        key=f"grid_editor_{st.session_state.grid_version}",
        use_container_width=True,
        height=400
    )
    
    # Update grid if changed
    if not np.array_equal(edited_grid, st.session_state.grid):
        st.session_state.grid = edited_grid
        st.session_state.path = None # Reset path on grid change
        st.rerun()

