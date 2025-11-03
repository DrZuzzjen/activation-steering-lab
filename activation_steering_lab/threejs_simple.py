"""
Simple Three.js integration using gr.HTML
This embeds Three.js directly into the Gradio app without custom component complexity.
"""

def create_threejs_html(activation_data):
    """Generate HTML with embedded Three.js visualization."""
    
    # Convert activation data to JSON string for embedding
    import json
    data_json = json.dumps(activation_data)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js"></script>
        <style>
            #brain-canvas {{
                width: 100%;
                height: 800px;
                background: rgb(10, 10, 10);
                display: block;
            }}
            #info-panel {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 100;
            }}
        </style>
    </head>
    <body style="margin: 0; padding: 0;">
        <div style="position: relative;">
            <canvas id="brain-canvas"></canvas>
            <div id="info-panel">
                <div><strong>Concept:</strong> <span id="concept-name"></span></div>
                <div><strong>Injection Layer:</strong> <span id="injection-layer"></span></div>
                <div><strong>Peak Layer:</strong> <span id="peak-layer"></span></div>
                <div><strong>🖱️ Mouse:</strong> Drag to rotate, Scroll to zoom</div>
            </div>
        </div>
        
        <script>
            const activationData = {data_json};
            
            // Scene setup
            const canvas = document.getElementById('brain-canvas');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            const camera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 1000);
            camera.position.set(50, 30, 50);
            camera.lookAt(0, 15, 0);
            
            const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            renderer.shadowMap.enabled = true;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
            directionalLight.position.set(10, 20, 10);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Simple orbit controls
            let isDragging = false;
            let previousMousePosition = {{ x: 0, y: 0 }};
            let rotation = {{ x: 0, y: 0 }};
            
            canvas.addEventListener('mousedown', (e) => {{ isDragging = true; }});
            canvas.addEventListener('mouseup', () => {{ isDragging = false; }});
            canvas.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    const deltaX = e.offsetX - previousMousePosition.x;
                    const deltaY = e.offsetY - previousMousePosition.y;
                    rotation.y += deltaX * 0.01;
                    rotation.x += deltaY * 0.01;
                }}
                previousMousePosition = {{ x: e.offsetX, y: e.offsetY }};
            }});
            
            canvas.addEventListener('wheel', (e) => {{
                e.preventDefault();
                camera.position.z += e.deltaY * 0.1;
            }});
            
            // Create layers
            const numLayers = activationData.metadata.num_layers;
            const numRegions = activationData.metadata.downsampled_regions;
            const injectionLayer = activationData.metadata.injection_layer;
            
            const gridSize = Math.ceil(Math.sqrt(numRegions));
            const regionSize = 0.8;
            const layerSpacing = 2;
            
            // Find max intensity for normalization
            let maxIntensity = 0;
            for (let i = 0; i < numLayers; i++) {{
                const acts = activationData.difference_activations[i.toString()];
                if (acts) {{
                    maxIntensity = Math.max(maxIntensity, ...acts);
                }}
            }}
            
            // Color scale function
            function getActivationColor(normalized) {{
                if (normalized < 0.3) {{
                    const t = normalized / 0.3;
                    return new THREE.Color().lerpColors(
                        new THREE.Color(0.235, 0.235, 0.235),
                        new THREE.Color(0.392, 0.0, 0.0),
                        t
                    );
                }} else if (normalized < 0.5) {{
                    const t = (normalized - 0.3) / 0.2;
                    return new THREE.Color().lerpColors(
                        new THREE.Color(0.392, 0.0, 0.0),
                        new THREE.Color(0.784, 0.0, 0.0),
                        t
                    );
                }} else if (normalized < 0.7) {{
                    const t = (normalized - 0.5) / 0.2;
                    return new THREE.Color().lerpColors(
                        new THREE.Color(0.784, 0.0, 0.0),
                        new THREE.Color(1.0, 0.392, 0.0),
                        t
                    );
                }} else if (normalized < 0.85) {{
                    const t = (normalized - 0.7) / 0.15;
                    return new THREE.Color().lerpColors(
                        new THREE.Color(1.0, 0.392, 0.0),
                        new THREE.Color(1.0, 0.784, 0.0),
                        t
                    );
                }} else {{
                    const t = (normalized - 0.85) / 0.15;
                    return new THREE.Color().lerpColors(
                        new THREE.Color(1.0, 0.784, 0.0),
                        new THREE.Color(1.0, 1.0, 1.0),
                        t
                    );
                }}
            }}
            
            // Create layer meshes
            for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {{
                const layerGroup = new THREE.Group();
                const activations = activationData.difference_activations[layerIdx.toString()];
                
                if (!activations) continue;
                
                for (let i = 0; i < numRegions; i++) {{
                    const row = Math.floor(i / gridSize);
                    const col = i % gridSize;
                    
                    const intensity = activations[i] || 0;
                    const normalized = intensity / maxIntensity;
                    
                    const geometry = new THREE.BoxGeometry(regionSize * 0.9, 0.5, regionSize * 0.9);
                    const color = getActivationColor(normalized);
                    const material = new THREE.MeshPhongMaterial({{
                        color: color,
                        emissive: color,
                        emissiveIntensity: normalized * 0.5,
                        transparent: true,
                        opacity: 0.8 + normalized * 0.2
                    }});
                    
                    const cube = new THREE.Mesh(geometry, material);
                    cube.position.set(
                        (col - gridSize / 2) * regionSize,
                        0,
                        (row - gridSize / 2) * regionSize
                    );
                    cube.castShadow = true;
                    cube.receiveShadow = true;
                    
                    layerGroup.add(cube);
                }}
                
                // Mark injection layer
                if (layerIdx === injectionLayer) {{
                    const outlineGeometry = new THREE.PlaneGeometry(gridSize * regionSize, gridSize * regionSize);
                    const outlineMaterial = new THREE.MeshBasicMaterial({{
                        color: 0x00ffff,
                        side: THREE.DoubleSide,
                        transparent: true,
                        opacity: 0.3
                    }});
                    const outline = new THREE.Mesh(outlineGeometry, outlineMaterial);
                    outline.rotation.x = -Math.PI / 2;
                    layerGroup.add(outline);
                }}
                
                layerGroup.position.y = layerIdx * layerSpacing;
                scene.add(layerGroup);
            }}
            
            // Update info panel
            document.getElementById('concept-name').textContent = activationData.metadata.concept_name;
            document.getElementById('injection-layer').textContent = activationData.metadata.injection_layer;
            document.getElementById('peak-layer').textContent = activationData.peak_activation_layer;
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                
                // Apply rotation
                scene.rotation.y = rotation.y;
                scene.rotation.x = rotation.x;
                
                renderer.render(scene, camera);
            }}
            animate();
            
            // Handle resize
            window.addEventListener('resize', () => {{
                camera.aspect = canvas.clientWidth / canvas.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    
    return html
