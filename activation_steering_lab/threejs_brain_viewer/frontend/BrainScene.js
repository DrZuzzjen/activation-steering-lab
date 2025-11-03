import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createBrainScene(canvas, activationData) {
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a); // Very dark like MRI

    // Camera with perspective
    const camera = new THREE.PerspectiveCamera(
        60,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
    );
    camera.position.set(50, 30, 50);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.shadowMap.enabled = true;

    // Orbit controls for rotation/zoom
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting (critical for depth perception)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Add spotlight on injection layer
    const spotLight = new THREE.SpotLight(0x00ffff, 0.5);
    spotLight.position.set(0, 50, 0);
    scene.add(spotLight);

    // Create layer meshes
    const layers = createLayerMeshes(activationData);
    layers.forEach(layer => scene.add(layer));

    // Animation loop
    let animationFrameId;
    function animate() {
        animationFrameId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    function handleResize() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }
    window.addEventListener('resize', handleResize);

    return {
        updateActivations: (newData) => {
            // Remove old layers
            layers.forEach(layer => scene.remove(layer));
            layers.length = 0;
            
            // Create new layers with new data
            const newLayers = createLayerMeshes(newData);
            newLayers.forEach(layer => {
                scene.add(layer);
                layers.push(layer);
            });
        },
        dispose: () => {
            window.removeEventListener('resize', handleResize);
            cancelAnimationFrame(animationFrameId);
            renderer.dispose();
            controls.dispose();
        }
    };
}

function createLayerMeshes(activationData) {
    const layers = [];
    
    if (!activationData || !activationData.metadata) {
        console.warn("No activation data provided");
        return layers;
    }

    const numLayers = activationData.metadata.num_layers;
    const numRegions = activationData.metadata.downsampled_regions;
    const injectionLayer = activationData.metadata.injection_layer;

    const regionSize = 0.8; // Size of each region cell
    const layerSpacing = 2; // Vertical spacing between layers

    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        // Create grid of cubes for this layer
        const layerGroup = new THREE.Group();

        const gridSize = Math.ceil(Math.sqrt(numRegions));
        const activations = activationData.difference_activations[layerIdx.toString()];

        if (!activations) {
            console.warn(`No activations for layer ${layerIdx}`);
            continue;
        }

        // Calculate intensity range for normalization
        const maxIntensity = Math.max(...Object.values(activationData.difference_activations).flat());
        const minIntensity = 0;

        for (let i = 0; i < numRegions; i++) {
            const row = Math.floor(i / gridSize);
            const col = i % gridSize;

            const intensity = activations[i] || 0;
            const normalizedIntensity = (intensity - minIntensity) / (maxIntensity - minIntensity);

            // Create cube for this region
            const geometry = new THREE.BoxGeometry(regionSize * 0.9, 0.5, regionSize * 0.9);
            const color = getActivationColor(normalizedIntensity);
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: normalizedIntensity * 0.5,
                transparent: true,
                opacity: 0.8 + normalizedIntensity * 0.2
            });

            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(
                (col - gridSize / 2) * regionSize,
                0,
                (row - gridSize / 2) * regionSize
            );
            cube.castShadow = true;
            cube.receiveShadow = true;

            layerGroup.add(cube);
        }

        // Position layer vertically
        layerGroup.position.y = layerIdx * layerSpacing;

        // Mark injection layer
        if (layerIdx === injectionLayer) {
            // Add cyan outline
            const outlineGeometry = new THREE.PlaneGeometry(gridSize * regionSize, gridSize * regionSize);
            const outlineMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ffff,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.3
            });
            const outline = new THREE.Mesh(outlineGeometry, outlineMaterial);
            outline.rotation.x = -Math.PI / 2;
            layerGroup.add(outline);
        }

        layers.push(layerGroup);
    }

    return layers;
}

function getActivationColor(normalizedIntensity) {
    // Color scale matching Plotly visualization
    // 0.0 -> dark gray, 0.3 -> dark red, 0.5 -> red, 0.7 -> orange, 0.85 -> yellow, 1.0 -> white
    
    if (normalizedIntensity < 0.3) {
        // Dark gray to dark red
        const t = normalizedIntensity / 0.3;
        return new THREE.Color().lerpColors(
            new THREE.Color(0.235, 0.235, 0.235), // dark gray
            new THREE.Color(0.392, 0.0, 0.0),      // dark red
            t
        );
    } else if (normalizedIntensity < 0.5) {
        // Dark red to red
        const t = (normalizedIntensity - 0.3) / 0.2;
        return new THREE.Color().lerpColors(
            new THREE.Color(0.392, 0.0, 0.0),  // dark red
            new THREE.Color(0.784, 0.0, 0.0),  // red
            t
        );
    } else if (normalizedIntensity < 0.7) {
        // Red to orange
        const t = (normalizedIntensity - 0.5) / 0.2;
        return new THREE.Color().lerpColors(
            new THREE.Color(0.784, 0.0, 0.0),  // red
            new THREE.Color(1.0, 0.392, 0.0),  // orange
            t
        );
    } else if (normalizedIntensity < 0.85) {
        // Orange to yellow
        const t = (normalizedIntensity - 0.7) / 0.15;
        return new THREE.Color().lerpColors(
            new THREE.Color(1.0, 0.392, 0.0),  // orange
            new THREE.Color(1.0, 0.784, 0.0),  // yellow
            t
        );
    } else {
        // Yellow to white
        const t = (normalizedIntensity - 0.85) / 0.15;
        return new THREE.Color().lerpColors(
            new THREE.Color(1.0, 0.784, 0.0),  // yellow
            new THREE.Color(1.0, 1.0, 1.0),    // white
            t
        );
    }
}
