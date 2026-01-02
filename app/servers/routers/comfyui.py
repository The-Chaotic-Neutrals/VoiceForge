"""
ComfyUI Router - ComfyUI integration endpoints.

Handles:
- /api/comfyui/status - Connection status
- /api/comfyui/launch - Launch ComfyUI
- /api/comfyui/workflows - List workflows
- /api/comfyui/execute - Execute workflow
- /api/comfyui/history - Get execution history
"""

import asyncio
import json
import os
import subprocess
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

# Import common (sets up sys.path)
from .common import verify_auth, APP_DIR

from util.clients import get_shared_session


router = APIRouter(tags=["ComfyUI"])

# ComfyUI configuration
COMFYUI_DIR = os.path.join(APP_DIR, "comfyui", "ComfyUI")
DEFAULT_COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# Current ComfyUI URL (can be updated by the user)
_current_comfyui_url: str = DEFAULT_COMFYUI_URL

# Process tracking
_comfyui_process: Optional[subprocess.Popen] = None


def get_comfyui_url() -> str:
    """Get the current ComfyUI URL."""
    return _current_comfyui_url


def set_comfyui_url(url: str) -> None:
    """Set the current ComfyUI URL."""
    global _current_comfyui_url
    _current_comfyui_url = url.rstrip('/')


@router.get("/api/comfyui/status")
async def get_comfyui_status(url: Optional[str] = None, _: bool = Depends(verify_auth)):
    """Check ComfyUI connection status."""
    session = get_shared_session()
    
    # Use provided URL or current URL
    comfyui_url = url.rstrip('/') if url else get_comfyui_url()
    
    # Update the stored URL if one was provided
    if url:
        set_comfyui_url(url)
    
    try:
        response = session.get(f"{comfyui_url}/system_stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            
            # Get node info
            try:
                nodes_response = session.get(f"{comfyui_url}/object_info", timeout=10)
                node_count = len(nodes_response.json()) if nodes_response.status_code == 200 else 0
            except:
                node_count = 0
            
            # Get queue status
            try:
                queue_response = session.get(f"{comfyui_url}/queue", timeout=5)
                queue_info = queue_response.json() if queue_response.status_code == 200 else {}
            except:
                queue_info = {}
            
            return {
                "connected": True,
                "url": comfyui_url,
                "system_stats": stats,
                "node_count": node_count,
                "queue": queue_info,
            }
        else:
            return {
                "connected": False,
                "url": comfyui_url,
                "error": f"HTTP {response.status_code}",
            }
    except Exception as e:
        return {
            "connected": False,
            "url": comfyui_url,
            "error": str(e),
        }


@router.post("/api/comfyui/launch")
async def launch_comfyui(
    request: Request,
    _: bool = Depends(verify_auth)
):
    """Launch ComfyUI process."""
    global _comfyui_process
    
    comfyui_url = get_comfyui_url()
    
    # Check if already running
    status = await get_comfyui_status(None, _)
    if status.get("connected"):
        return {"success": True, "status": "already_running", "url": comfyui_url}
    
    # Get path from request body
    try:
        data = await request.json()
        comfyui_path = data.get("path", "").strip()
    except:
        comfyui_path = ""
    
    # Find launch script - check multiple locations
    launch_script = None
    search_paths = []
    
    if comfyui_path:
        # User-provided path - look for run_nvidia_gpu.bat there
        search_paths = [
            os.path.join(comfyui_path, "run_nvidia_gpu.bat"),
            os.path.join(comfyui_path, "ComfyUI", "run_nvidia_gpu.bat"),
        ]
    
    # Also check default app location
    search_paths.append(os.path.join(APP_DIR, "comfyui", "run_nvidia_gpu.bat"))
    
    for path in search_paths:
        if os.path.exists(path):
            launch_script = path
            break
    
    if not launch_script:
        return {
            "success": False, 
            "error": f"Could not find run_nvidia_gpu.bat. Searched: {', '.join(search_paths)}"
        }
    
    try:
        # Launch in a new console window (don't capture output - let it run in its own window)
        if os.name == 'nt':
            # On Windows, use start command to open in new window
            _comfyui_process = subprocess.Popen(
                f'start "ComfyUI" /D "{os.path.dirname(launch_script)}" "{launch_script}"',
                shell=True,
            )
        else:
            # On Linux/Mac
            _comfyui_process = subprocess.Popen(
                [launch_script],
                cwd=os.path.dirname(launch_script),
                start_new_session=True,
            )
        
        return {
            "success": True, 
            "status": "launching", 
            "url": comfyui_url, 
            "message": f"ComfyUI is starting from {launch_script}..."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to launch ComfyUI: {str(e)}"}


@router.get("/api/comfyui/workflows")
async def list_workflows(path: Optional[str] = None, _: bool = Depends(verify_auth)):
    """List available ComfyUI workflows."""
    workflows = []
    session = get_shared_session()
    comfyui_url = get_comfyui_url()
    
    # Try to get from ComfyUI userdata API (if ComfyUI is running)
    try:
        response = session.get(
            f"{comfyui_url}/userdata?dir=workflows&recurse=true&split=false&full_info=true",
            timeout=10
        )
        if response.status_code == 200:
            for item in response.json():
                if item.get("path", "").endswith(".json"):
                    workflows.append({
                        "name": os.path.splitext(os.path.basename(item["path"]))[0],
                        "path": item["path"],
                        "source": "comfyui_userdata",
                    })
    except:
        pass
    
    # If a ComfyUI path is provided, look for workflows in the filesystem
    if path:
        # Common workflow locations in ComfyUI installations
        workflow_dirs = [
            os.path.join(path, "user", "default", "workflows"),  # New ComfyUI structure
            os.path.join(path, "workflows"),                      # Direct workflows folder
            os.path.join(path, "ComfyUI", "user", "default", "workflows"),  # Nested ComfyUI
            os.path.join(path, "ComfyUI", "workflows"),           # Nested workflows
        ]
        
        seen_paths = set()
        for workflow_dir in workflow_dirs:
            if os.path.exists(workflow_dir) and os.path.isdir(workflow_dir):
                # Walk the directory tree to find all .json files
                for root, dirs, files in os.walk(workflow_dir):
                    for f in files:
                        if f.endswith('.json'):
                            full_path = os.path.join(root, f)
                            if full_path not in seen_paths:
                                seen_paths.add(full_path)
                                # Get relative path for display
                                rel_path = os.path.relpath(full_path, workflow_dir)
                                workflows.append({
                                    "name": os.path.splitext(f)[0],
                                    "path": full_path,
                                    "source": "comfyui_fs",
                                    "rel_path": rel_path,
                                })
    
    # Also check local app workflows directory
    local_workflows_dir = os.path.join(APP_DIR, "comfyui", "workflows")
    if os.path.exists(local_workflows_dir):
        for f in os.listdir(local_workflows_dir):
            if f.endswith('.json'):
                workflows.append({
                    "name": os.path.splitext(f)[0],
                    "path": os.path.join(local_workflows_dir, f),
                    "source": "local",
                })
    
    # Check bundled ComfyUI's user workflows directory
    bundled_comfyui_workflows = os.path.join(APP_DIR, "comfyui", "ComfyUI", "user", "default", "workflows")
    if os.path.exists(bundled_comfyui_workflows):
        seen_names = {wf["name"] for wf in workflows}  # Avoid duplicates from userdata API
        for root, dirs, files in os.walk(bundled_comfyui_workflows):
            for f in files:
                if f.endswith('.json'):
                    name = os.path.splitext(f)[0]
                    if name not in seen_names:
                        full_path = os.path.join(root, f)
                        workflows.append({
                            "name": name,
                            "path": full_path,
                            "source": "local",
                        })
                        seen_names.add(name)
    
    return {"workflows": workflows}


@router.get("/api/comfyui/workflow/{workflow_name:path}")
async def get_workflow(
    workflow_name: str, 
    source: str = "local",
    path: Optional[str] = None,
    _: bool = Depends(verify_auth)
):
    """Get a specific workflow by name."""
    session = get_shared_session()
    comfyui_url = get_comfyui_url()
    
    workflow = None
    
    # If a direct filesystem path is provided, use it
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading workflow: {str(e)}")
    
    # Try local app workflows directory
    if not workflow:
        local_path = os.path.join(APP_DIR, "comfyui", "workflows", f"{workflow_name}.json")
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            source = "local"
    
    # Try ComfyUI userdata API
    if not workflow:
        try:
            import urllib.parse
            encoded_path = urllib.parse.quote(f"workflows/{workflow_name}.json", safe='')
            response = session.get(f"{comfyui_url}/userdata/{encoded_path}", timeout=10)
            if response.status_code == 200:
                workflow = response.json()
                source = "comfyui_userdata"
        except:
            pass
    
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found")
    
    # Get object_info from ComfyUI to get proper input definitions
    object_info = None
    try:
        response = session.get(f"{comfyui_url}/object_info", timeout=15)
        if response.status_code == 200:
            object_info = response.json()
    except:
        pass
    
    return _parse_workflow_response(workflow_name, workflow, source, object_info)


def _parse_workflow_response(name: str, workflow: dict, source: str, object_info: dict = None) -> dict:
    """Parse a workflow and extract useful information for the UI."""
    inputs = []
    
    # Handle both API format and GUI format workflows
    nodes = {}
    is_gui_format = False
    
    # Check if this is GUI format (has "nodes" array at root or under "workflow")
    if "nodes" in workflow and isinstance(workflow.get("nodes"), list):
        # GUI format at root level
        is_gui_format = True
        gui_nodes = workflow.get("nodes", [])
    elif "workflow" in workflow and isinstance(workflow.get("workflow", {}).get("nodes"), list):
        # GUI format nested under "workflow" key
        is_gui_format = True
        gui_nodes = workflow.get("workflow", {}).get("nodes", [])
    else:
        gui_nodes = []
    
    if is_gui_format:
        # GUI format - nodes array with widgets_values as positional array
        for gui_node in gui_nodes:
            node_id = str(gui_node.get("id", ""))
            class_type = gui_node.get("type", "")
            node_title = gui_node.get("title", class_type)
            widgets_values = gui_node.get("widgets_values", [])
            node_inputs_def = gui_node.get("inputs", [])
            
            if not node_id or not class_type:
                continue
            
            # Skip nodes with no widgets
            if not widgets_values:
                nodes[node_id] = {
                    "class_type": class_type,
                    "inputs": {},
                    "_meta": {"title": node_title}
                }
                continue
            
            # Extract widget inputs from the node's inputs array (items with "widget" property)
            # IMPORTANT: Skip widgets that have an active link - they're controlled by connected nodes
            # These are in the correct order to match widgets_values
            widget_inputs = [inp for inp in node_inputs_def if inp.get("widget")]
            
            # Build a set of input names that have active links (connected to other nodes)
            linked_inputs = {inp.get("name") for inp in node_inputs_def if inp.get("link") is not None}
            
            # Build named inputs by mapping widgets_values to widget names
            # Handle hidden widgets like control_after_generate that appear after seed/INT inputs
            named_inputs = {}
            widget_defs_map = {}
            
            if widget_inputs:
                # Normal case: we have widget definitions in inputs array
                widget_idx = 0
                value_idx = 0
                
                while widget_idx < len(widget_inputs) and value_idx < len(widgets_values):
                    widget_def = widget_inputs[widget_idx]
                    widget_name = widget_def.get("name", widget_def.get("widget", {}).get("name", f"widget_{widget_idx}"))
                    widget_type = widget_def.get("type", "STRING")
                    
                    # Only extract value if this input is NOT connected to another node
                    # Connected inputs are controlled by the source node, not editable here
                    if widget_name not in linked_inputs:
                        named_inputs[widget_name] = widgets_values[value_idx]
                        widget_defs_map[widget_name] = widget_def
                    
                    value_idx += 1
                    
                    # Check for hidden control_after_generate widget after seed/INT widgets
                    # ComfyUI adds this hidden widget after certain inputs
                    if widget_type == "INT" and widget_name in ("seed", "noise_seed"):
                        # Skip the control_after_generate value if present
                        if value_idx < len(widgets_values) and isinstance(widgets_values[value_idx], str) and widgets_values[value_idx] in ("fixed", "increment", "decrement", "randomize"):
                            value_idx += 1  # Skip this hidden widget value
                    
                    widget_idx += 1
            else:
                # No widget definitions in inputs (e.g., Seed rgthree node)
                # Try to use object_info if available, otherwise use generic names
                if object_info and class_type in object_info:
                    node_info = object_info[class_type]
                    widget_names = []
                    # Get widget names from object_info
                    for inp_name, inp_def in node_info.get("input", {}).get("required", {}).items():
                        type_info = inp_def[0] if inp_def else None
                        if isinstance(type_info, list) or type_info in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                            widget_names.append(inp_name)
                    for inp_name, inp_def in node_info.get("input", {}).get("optional", {}).items():
                        type_info = inp_def[0] if inp_def else None
                        if isinstance(type_info, list) or type_info in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                            widget_names.append(inp_name)
                    
                    for i, widget_name in enumerate(widget_names):
                        if i < len(widgets_values) and widgets_values[i] is not None:
                            named_inputs[widget_name] = widgets_values[i]
                else:
                    # Fallback: use generic names for non-null values
                    # Common pattern: first value is often "seed" or "value" for seed nodes
                    generic_names = ["seed", "control_after_generate", "action", "last_seed"]
                    for i, val in enumerate(widgets_values):
                        if val is not None:
                            name = generic_names[i] if i < len(generic_names) else f"value_{i}"
                            named_inputs[name] = val
                            # Add type hint based on value
                            if isinstance(val, int):
                                widget_defs_map[name] = {"type": "INT", "name": name}
                            elif isinstance(val, float):
                                widget_defs_map[name] = {"type": "FLOAT", "name": name}
                            elif isinstance(val, str):
                                widget_defs_map[name] = {"type": "STRING", "name": name}
            
            # Store the node with extracted inputs and original input definitions for type info
            nodes[node_id] = {
                "class_type": class_type,
                "inputs": named_inputs,
                "_meta": {"title": node_title},
                "_widget_defs": widget_defs_map if widget_defs_map else {inp.get("name"): inp for inp in widget_inputs}
            }
    else:
        # API format - nodes are at root level with numeric/string keys
        nodes = {k: v for k, v in workflow.items() if k not in ("extra", "version", "last_node_id", "last_link_id", "links", "groups", "config") and isinstance(v, dict) and "class_type" in v}
    
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
            
        class_type = node.get("class_type", "")
        node_title = node.get("_meta", {}).get("title", class_type)
        node_inputs = node.get("inputs", {})
        widget_defs = node.get("_widget_defs", {})  # GUI format type info
        
        # Get input definitions from object_info if available
        input_defs = {}
        if object_info and class_type in object_info:
            node_info = object_info[class_type]
            # required inputs
            for inp_name, inp_def in node_info.get("input", {}).get("required", {}).items():
                input_defs[inp_name] = {"def": inp_def, "required": True}
            # optional inputs
            for inp_name, inp_def in node_info.get("input", {}).get("optional", {}).items():
                input_defs[inp_name] = {"def": inp_def, "required": False}
        
        for input_name, input_value in node_inputs.items():
            # Skip linked inputs (arrays referencing other nodes)
            if isinstance(input_value, list) and len(input_value) == 2:
                # This is a link to another node [node_id, output_index]
                continue
            
            # Get definition for this input from object_info or GUI widget defs
            inp_info = input_defs.get(input_name, {})
            inp_def = inp_info.get("def", [])
            required = inp_info.get("required", False)
            gui_widget_def = widget_defs.get(input_name, {})
            gui_type = gui_widget_def.get("type", "")
            
            # Determine type and constraints
            input_type = "string"
            options = None
            min_val = None
            max_val = None
            step_val = None
            multiline = False
            
            # First try object_info for detailed type info
            if inp_def:
                type_info = inp_def[0] if inp_def else None
                constraints = inp_def[1] if len(inp_def) > 1 else {}
                
                if isinstance(type_info, list):
                    # Enum/combo type
                    input_type = "enum"
                    options = type_info
                elif type_info == "INT":
                    input_type = "int"
                    if isinstance(constraints, dict):
                        min_val = constraints.get("min")
                        max_val = constraints.get("max")
                        step_val = constraints.get("step", 1)
                elif type_info == "FLOAT":
                    input_type = "float"
                    if isinstance(constraints, dict):
                        min_val = constraints.get("min")
                        max_val = constraints.get("max")
                        step_val = constraints.get("step", 0.01)
                elif type_info == "STRING":
                    input_type = "string"
                    if isinstance(constraints, dict):
                        multiline = constraints.get("multiline", False)
                elif type_info == "BOOLEAN":
                    input_type = "boolean"
            # Then try GUI widget type info
            elif gui_type:
                if gui_type == "INT":
                    input_type = "int"
                elif gui_type == "FLOAT":
                    input_type = "float"
                elif gui_type == "STRING":
                    input_type = "string"
                    # Check if it's likely a prompt
                    if input_name in ("text", "prompt", "positive", "negative") or (isinstance(input_value, str) and len(input_value) > 50):
                        multiline = True
                elif gui_type == "BOOLEAN":
                    input_type = "boolean"
                elif gui_type == "COMBO":
                    input_type = "enum"
                    # We don't have options without object_info, just show current value
                    options = [input_value] if input_value else []
            # Finally infer from value
            else:
                if isinstance(input_value, bool):
                    input_type = "boolean"
                elif isinstance(input_value, int):
                    input_type = "int"
                elif isinstance(input_value, float):
                    input_type = "float"
                elif isinstance(input_value, str):
                    input_type = "string"
                    # Check if it's likely a prompt
                    if input_name in ("text", "prompt", "positive", "negative") or len(str(input_value)) > 50:
                        multiline = True
            
            # Create label from input name
            label = input_name.replace("_", " ").title()
            
            input_data = {
                "node_id": node_id,
                "node_title": node_title,
                "class_type": class_type,
                "param": input_name,
                "label": label,
                "type": input_type,
                "value": input_value,
                "required": required,
                "multiline": multiline,
            }
            
            if options is not None:
                input_data["options"] = options
            if min_val is not None:
                input_data["min"] = min_val
            if max_val is not None:
                input_data["max"] = max_val
            if step_val is not None:
                input_data["step"] = step_val
            
            inputs.append(input_data)
    
    # Convert GUI format to API format for execution
    api_workflow = _convert_to_api_format(workflow, nodes) if is_gui_format else workflow
    
    return {
        "name": name,
        "source": source,
        "workflow": api_workflow,
        "inputs": inputs,
    }


def _convert_to_api_format(original_workflow: dict, parsed_nodes: dict) -> dict:
    """Convert a GUI format workflow to API format for execution."""
    api_workflow = {}
    
    # Build link map first
    link_map = {}
    if "links" in original_workflow:
        links = original_workflow.get("links", [])
        for link in links:
            if len(link) >= 5:
                link_id, from_node, from_slot = link[0], link[1], link[2]
                link_map[link_id] = [str(from_node), from_slot]
    
    # Process ALL nodes from the original GUI workflow, not just parsed ones
    gui_nodes = original_workflow.get("nodes", [])
    for gui_node in gui_nodes:
        node_id = str(gui_node.get("id", ""))
        class_type = gui_node.get("type", "")
        
        if not node_id or not class_type:
            continue
        
        # Start with parsed inputs (widget values) if available
        node_inputs = {}
        if node_id in parsed_nodes:
            node_inputs = dict(parsed_nodes[node_id].get("inputs", {}))
        
        # Add all linked inputs from the GUI node
        node_inputs_def = gui_node.get("inputs", [])
        for inp_def in node_inputs_def:
            inp_name = inp_def.get("name", "")
            link_id = inp_def.get("link")
            
            if link_id is not None and link_id in link_map:
                # This input is connected to another node - use the link
                node_inputs[inp_name] = link_map[link_id]
        
        api_workflow[node_id] = {
            "class_type": class_type,
            "inputs": node_inputs
        }
        
        # Preserve _meta/title
        node_title = gui_node.get("title", class_type)
        api_workflow[node_id]["_meta"] = {"title": node_title}
    
    return api_workflow


@router.post("/api/comfyui/workflow/upload")
async def upload_workflow(
    request: Request,
    _: bool = Depends(verify_auth)
):
    """Upload a workflow file."""
    try:
        workflow_data = await request.json()
        
        name = workflow_data.get("name", "untitled")
        workflow = workflow_data.get("workflow", workflow_data)
        
        # Save locally
        workflows_dir = os.path.join(APP_DIR, "comfyui", "workflows")
        os.makedirs(workflows_dir, exist_ok=True)
        
        path = os.path.join(workflows_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)
        
        return {"status": "uploaded", "name": name, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/comfyui/object_info")
async def get_object_info(_: bool = Depends(verify_auth)):
    """Get ComfyUI node definitions."""
    session = get_shared_session()
    comfyui_url = get_comfyui_url()
    
    try:
        response = session.get(f"{comfyui_url}/object_info", timeout=30)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI not available: {str(e)}")


@router.post("/api/comfyui/execute")
async def execute_workflow(
    request: Request,
    _: bool = Depends(verify_auth)
):
    """
    Execute a ComfyUI workflow.
    
    Body should contain either:
    - prompt: The workflow prompt to execute (API format)
    - workflow: The workflow to execute (alternative key name)
    - workflow_name: Name of saved workflow to execute
    """
    import random
    
    session = get_shared_session()
    comfyui_url = get_comfyui_url()
    
    try:
        data = await request.json()
        
        # Accept both "prompt" and "workflow" keys
        prompt = data.get("prompt") or data.get("workflow")
        
        # If workflow_name provided, load it
        if not prompt and data.get("workflow_name"):
            workflow_response = await get_workflow(data["workflow_name"], "local", None, _)
            prompt = workflow_response.get("workflow")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt or workflow_name provided")
        
        # Handle special seed values: -1 means random
        # ComfyUI doesn't accept negative seeds, so we generate a random one
        # Note: Different nodes have different max seed values:
        # - Standard ComfyUI: up to 2^64-1
        # - rgthree Seed: up to 2^50 (1125899906842624)
        # Use a safe max that works for all nodes
        SAFE_MAX_SEED = 1125899906842624  # 2^50, compatible with rgthree
        
        for node_id, node in prompt.items():
            if isinstance(node, dict) and "inputs" in node:
                inputs = node["inputs"]
                for key in ["seed", "noise_seed"]:
                    if key in inputs and isinstance(inputs[key], (int, float)):
                        current_val = inputs[key]
                        # Handle negative values (means random)
                        if current_val < 0:
                            new_seed = random.randint(0, SAFE_MAX_SEED)
                            inputs[key] = new_seed
                            print(f"[ComfyUI] Generated random seed for node {node_id}.{key}: {new_seed}")
                        # Also clamp seeds that are too large
                        elif current_val > SAFE_MAX_SEED:
                            inputs[key] = current_val % SAFE_MAX_SEED
                            print(f"[ComfyUI] Clamped large seed for node {node_id}.{key}: {current_val} -> {inputs[key]}")
        
        # Log the workflow being submitted (for debugging)
        print(f"[ComfyUI] Submitting workflow with {len(prompt)} nodes")
        
        # Submit to ComfyUI
        response = session.post(
            f"{comfyui_url}/prompt",
            json={"prompt": prompt, "client_id": data.get("client_id", "voiceforge")},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/comfyui/history/{prompt_id}")
async def get_history(prompt_id: str, _: bool = Depends(verify_auth)):
    """Get execution history for a prompt."""
    session = get_shared_session()
    comfyui_url = get_comfyui_url()
    
    try:
        response = session.get(f"{comfyui_url}/history/{prompt_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/api/comfyui/view")
async def view_output(
    filename: str,
    subfolder: str = "",
    type: str = "output",
    _: bool = Depends(verify_auth)
):
    """View/download output from ComfyUI."""
    session = get_shared_session()
    comfyui_url = get_comfyui_url()
    
    try:
        params = {"filename": filename, "type": type}
        if subfolder:
            params["subfolder"] = subfolder
        
        response = session.get(f"{comfyui_url}/view", params=params, timeout=30)
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type=response.headers.get("Content-Type", "application/octet-stream"),
            )
        raise HTTPException(status_code=response.status_code, detail=response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
