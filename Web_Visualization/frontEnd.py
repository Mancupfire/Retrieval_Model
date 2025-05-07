import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import requests
from collections import defaultdict

# Constants for positioning
GRAPH_WIDTH  = 800   # should match your CSS/container width
GRAPH_HEIGHT = 600   # should match internal_styles["graph"]["height"]
LEFT_X       = 100   # x-position for keywords
RIGHT_X      = GRAPH_WIDTH - 100  # x-position for restaurants
MIN_RT_SPACING = 50

# Initialize the Dash app
app = dash.Dash(__name__)

# ------------------------------------------------------------------------------
# Internal CSS Styling with Flexbox Layout
# ------------------------------------------------------------------------------
internal_styles = {
    "container": {
        "fontFamily": "Arial, sans-serif",
        "padding": "20px",
        "maxWidth": "1000px",
        "margin": "0 auto"
    },
    "header": {
        "textAlign": "center",
        "color": "#333",
        "marginBottom": "20px"
    },
    # A generic row style for horizontally aligned items
    "row": {
        "display": "flex",
        "flexWrap": "wrap",
        "alignItems": "center",
        "justifyContent": "space-between",
        "gap": "10px",
        "marginBottom": "20px"
    },
    # Each dropdown in the row can expand/shrink as needed
    "dropdown": {
        "flex": "1",
        "minWidth": "150px"
    },
    # Specifically for the max-restaurant dropdown (if you want a fixed width)
    "dropdownMaxRest": {
        "width": "200px"
    },
    # Container for search input and button
    "searchContainer": {
        "display": "flex",
        "alignItems": "center",
        "gap": "10px",
        "marginBottom": "20px"
    },
    "input": {
        "flex": "1",
        "padding": "10px",
        "border": "1px solid #ccc",
        "borderRadius": "5px"
    },
    "button": {
        "padding": "10px 20px",
        "backgroundColor": "#007BFF",
        "color": "#fff",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "pointer"
    },
    "graph": {
        "width": "100%",
        "height": "600px",
        "border": "1px solid #ddd",
        "borderRadius": "5px",
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
    },
    "popup": {
        "position": "absolute",
        "zIndex": 10,
        "backgroundColor": "#FFFFFF",
        "border": "1px solid #CCC",
        "borderRadius": "5px",
        "padding": "10px",
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
        "width": "200px",
        "transition": "opacity 0.3s ease-in-out",
    }
}

# ------------------------------------------------------------------------------
# Cytoscape Stylesheet
# ------------------------------------------------------------------------------
cytoscape_stylesheet = [
    {
    "selector": ".node-keyword", 
    "style": {
        "label": "data(label)", 
        "background-color": "#33FF57",
        "border-width": "0px",         # thickness of the border
        "border-style": "solid",       # solid line
        "border-color": "black",       # black border
        "shadow-blur": 6,              # how much the shadow spreads
        "shadow-color": "rgba(0,0,0,0.4)",  # semi‑transparent black
        "shadow-opacity": 0.6,         # overall shadow opacity
        "shadow-offset-x": 2,          # horizontal shadow offset
        "shadow-offset-y": 2   
        }
    },
    {
    "selector": ".node-restaurant", 
    "style": {
         "label": "data(label)", 
         "background-color": "#ffffd4",
         "border-width": "3px",         # thickness of the border
        "border-style": "solid",       # solid line
        "border-color": "black",       # black border
        "shadow-blur": 6,              # how much the shadow spreads
        "shadow-color": "rgba(0,0,0,0.4)",  # semi‑transparent black
        "shadow-opacity": 0.6,         # overall shadow opacity
        "shadow-offset-x": 2,          # horizontal shadow offset
        "shadow-offset-y": 2  
         }
    }
]

# ------------------------------------------------------------------------------
# App Layout
# ------------------------------------------------------------------------------
app.layout = html.Div(style=internal_styles["container"], children=[
    html.Div(
    className="background-shapes",
    children=[
        html.Div(className="shape circle1"),
        html.Div(className="shape circle2"),
        html.Div(className="shape circle3"),
        html.Div(className="shape circle4"),
    ]
    ),

    html.H1("Keyword-driven recommender system", style=internal_styles["header"]),

    # Row of 4 dropdowns for models
    html.Div(style=internal_styles["row"], children=[
        dcc.Dropdown(
            id="rec-model",
            options=[],  # Add your options here
            placeholder="Rec model",
            style=internal_styles["dropdown"]
        ),
        dcc.Dropdown(
            id="retrieval-model",
            options=[],  # Add your options here
            placeholder="Retrieval model",
            style=internal_styles["dropdown"]
        ),
        dcc.Dropdown(
            id="rerank-model",
            options=[],  # Add your options here
            placeholder="re-rank model",
            style=internal_styles["dropdown"]
        ),
        dcc.Dropdown(
            id="sort-by",
            options=[],  # Add your options here
            placeholder="sort by",
            style=internal_styles["dropdown"]
        )
    ]),

    # Dropdown to limit the number of restaurants per keyword
    html.Div(style=internal_styles["row"], children=[
        dcc.Dropdown(
            id="max-restaurant-dropdown",
            options=[{"label": str(i), "value": i} for i in [5, 10, 15, 20]], #Adjust for user input
            placeholder="Max Restaurants",
            value=5,  # Default to 10
            style=internal_styles["dropdownMaxRest"]
        )
    ]),

    # Search input + button in a single row
    html.Div(style=internal_styles["searchContainer"], children=[
        dcc.Input(
            id="keyword-input",
            type="text",
            placeholder="Enter keywords, separated by commas",
            style=internal_styles["input"]
        ),
        html.Button("Search", id="search-button", style=internal_styles["button"])
    ]),

    # Container for the popup
    html.Div(id="popup-container", style={"position": "relative"}),

    # Cytoscape graph
    cyto.Cytoscape(
        id="cytoscape-graph",
        layout={"name": "preset"},       # use preset positions
        stylesheet=cytoscape_stylesheet,
        style=internal_styles["graph"],
        elements=[]
    ),
    

    # ───── Legend Overlay ─────
    html.Div(
        style={
            "position": "absolute",
            "top": "20px", 
            "left": "20px",
            "backgroundColor": "rgba(255, 255, 255, 0.8)",
            "padding": "10px",
            "borderRadius": "5px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.15)",
            "zIndex": 10
        },
        children=[
            html.Div(style={"fontWeight": "bold", "marginBottom": "5px"}, children="Notes"),
            # 1 keyword
            html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                html.Div(style={
                    "width": "12px", "height": "12px",
                    "borderRadius": "50%",
                    "backgroundColor": "#ffffd4",
                    "marginRight": "6px"
                }),
                html.Span("1 keyword")
            ]),
            # 2 keywords
            html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                html.Div(style={
                    "width": "12px", "height": "12px",
                    "borderRadius": "50%",
                    "backgroundColor": "#fed98e",
                    "marginRight": "6px"
                }),
                html.Span("2 keywords")
            ]),
            # 3 keywords
            html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}, children=[
                html.Div(style={
                    "width": "12px", "height": "12px",
                    "borderRadius": "50%",
                    "backgroundColor": "#fe9929",
                    "marginRight": "6px"
                }),
                html.Span("3 keywords")
            ]),
            # 4+ keywords
            html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                html.Div(style={
                    "width": "12px", "height": "12px",
                    "borderRadius": "50%",
                    "backgroundColor": "#d95f0e",
                    "marginRight": "6px"
                }),
                html.Span("4+ keywords")
            ]),
        ]
    ),

    
    html.Div(
    id="sidebar",
    children=[
        # Close button
        html.Button(
            "×",
            id="close-button",
            n_clicks=0,
            style={
                "background": "none",
                "border": "none",
                "fontSize": "24px",
                "position": "absolute",
                "top": "10px",
                "right": "10px",
                "cursor": "pointer"
            }
        ),
        # Container for dynamic content
        html.Div(id="sidebar-content", style={"padding": "20px", "marginTop": "40px"})
    ],
    style={
        "position": "fixed",
        "top": 0,
        "right": "-25%",         # start off-screen
        "width": "25%",
        "height": "100%",
        "backgroundColor": "#FFFFFF",
        "boxShadow": "-2px 0 8px rgba(0,0,0,0.2)",
        "transition": "right 0.2s ease-out",
        "zIndex": 999
    }
    )
    
])

# ------------------------------------------------------------------------------
# compute_dynamic_yellow: color logic for multi-connection restaurants
# ------------------------------------------------------------------------------
def compute_dynamic_yellow(connection_count):
    """
    Returns a hex color based on the number of keyword connections.
    For connection_count == 2: #FFEC8B
    For 3: #FFD700
    For 4: #FFF68F
    For 5 or more: #FFF8DC
    Otherwise: #33FF57 (fallback)
    """
    if connection_count == 2:
        return "#fed98e"
    elif connection_count == 3:
        return "#fe9929"
    elif connection_count == 4:
        return "#d95f0e"
    elif connection_count >= 5:
        return "#993404"
    else:
        return "#33FF57"  # Fallback color

# ------------------------------------------------------------------------------
# Callback: Update the graph based on selected keywords & max restaurants
# ------------------------------------------------------------------------------
@app.callback(
    [Output("cytoscape-graph", "elements"),
     Output("keyword-input", "value")],
    [Input("search-button", "n_clicks")],
    [State("keyword-input", "value"),
     State("max-restaurant-dropdown", "value")]
)
def update_graph(n_clicks, keyword_input, max_rest):
    if not keyword_input:
        return [], ""

    # 1. Parse and sort keywords
    selected_keywords = sorted(kw.strip() for kw in keyword_input.split(","))

    # 2. Fetch edges
    resp = requests.get(
        "http://127.0.0.1:5000/get_restaurant_keywords",
        params={"keywords": selected_keywords}
    )
    edges = resp.json()

    # 3. Group & limit per keyword
    kw_to_edges = defaultdict(list)
    for e in edges:
        kw_to_edges[e["keyword"]].append(e)

    limited_edges = []
    for kw in selected_keywords:
        limited_edges.extend(kw_to_edges[kw][:max_rest])

    # 4. Unique, sorted restaurants
    restaurants = sorted({e["restaurant"] for e in limited_edges})

    # 5. Compute keyword spacing (unchanged)
    n_kw = len(selected_keywords)
    y_spacing_kw = GRAPH_HEIGHT / (n_kw + 1)

    # 6. Compute restaurant spacing with a minimum
    n_rt = len(restaurants)
    # base evenly‐divided spacing
    base_rt_spacing = GRAPH_HEIGHT / (n_rt + 1) if n_rt > 0 else 0
    # enforce minimum
    y_spacing_rt = max(base_rt_spacing, MIN_RT_SPACING)

    # 7. Count connections for dynamic yellow
    restaurant_counts = defaultdict(int)
    for e in limited_edges:
        restaurant_counts[e["restaurant"]] += 1

    # 8. Build Cytoscape elements
    elements = []

    # 8a. Keyword nodes
    for idx, kw in enumerate(selected_keywords, start=1):
        elements.append({
            "data": {"id": kw, "label": kw},
            "classes": "node-keyword",
            "position": {"x": LEFT_X, "y": idx * y_spacing_kw}
        })

    # 8b. Restaurant nodes
    for idx, rt in enumerate(restaurants, start=1):
        y = idx * y_spacing_rt
        count = restaurant_counts[rt]

        if count > 1:
            color = compute_dynamic_yellow(count)
            node = {
                "data": {"id": rt, "label": f"Restaurant {rt[:5]}"},
                "style": {
                    "label": f"Restaurant {rt[:5]}",
                    "background-color": color
                },
                "position": {"x": RIGHT_X, "y": y}
            }
        else:
            node = {
                "data": {"id": rt, "label": f"Restaurant {rt[:5]}"},
                "classes": "node-restaurant",
                "position": {"x": RIGHT_X, "y": y}
            }

        elements.append(node)

    # 8c. Edges
    for e in limited_edges:
        elements.append({
            "data": {"source": e["keyword"], "target": e["restaurant"]}
        })

    # 9. Return elements and restore input text
    return elements, ", ".join(selected_keywords)
# ------------------------------------------------------------------------------
# Callback: Display popup on restaurant node click
# ------------------------------------------------------------------------------
# 2) Replace your old popup callback with this sliding‐panel callback:

@app.callback(
    [
        Output("sidebar-content", "children"),
        Output("sidebar", "style")
    ],
    [
        Input("cytoscape-graph", "tapNode"),
        Input("cytoscape-graph", "tapBlank"),
        Input("close-button", "n_clicks")
    ],
    [State("keyword-input", "value"),
     State("sidebar", "style")]
)
def slide_panel(tap_node, tap_blank, close_clicks, current_keywords, style):
    ctx = dash.callback_context
    if not ctx.triggered:
        # No action yet
        return dash.no_update, style

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    new_style = style.copy()

    # 1) Click on a node → show panel
    if trigger == "cytoscape-graph" and tap_node:
        node_label = tap_node["data"]["label"]
        # Build the same content you had in your popup
        content = [
            html.H3(f"User comments for {node_label}", style={"margin": "0 0 10px", "fontSize": "20px"}),
            html.H3("User 1", style={"margin": "20px 0 10px", "fontSize": "16px"}),
            html.P(f"I love the style of {current_keywords} here, it is absolutely unique!", style={"fontSize": "14px"}),
            html.H3("User 2", style={"margin": "20px 0 10px", "fontSize": "16px"}),
            html.P(f"The {current_keywords} here is undoubtedly a must-to-try!", style={"fontSize": "14px"}),
            html.H3("User 3", style={"margin": "20px 0 10px", "fontSize": "16px"}),
            html.P(f"The chef definitely has a treasured {current_keywords}'s recipe!", style={"fontSize": "14px"})
        ]
        new_style["right"] = "0"  # slide in
        return content, new_style

    # 2) Click on blank space or close button → hide panel
    if trigger in ("cytoscape-graph", "close-button"):
        # hide regardless of where the click came from
        new_style["right"] = "-25%"
        return [], new_style

    # Fallback
    return dash.no_update, style


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
