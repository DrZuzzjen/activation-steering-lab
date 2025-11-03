"""
Quick integration test: Add export button to existing Gradio app
This lets you test the export functionality from the UI.
"""

print("""
🧪 INTEGRATION TEST PLAN
========================

To test Phase 1 in the existing app, you can:

1. Run the main app:
   ./run.sh

2. Go to "🔬 Visualization Lab" tab

3. Generate normal vs steered visualization

4. In the future (Phase 6), add a button like:
   "📥 Export to 3D View" 
   
   Which will call:
   ```python
   exporter = ThreeJSExporter()
   exporter.save_activations_for_threejs(
       normal_acts, steered_acts, tokens, metadata
   )
   ```

5. The exported JSON can then be loaded in Phase 2's Three.js viewer

CURRENT STATUS:
- ✅ Export module is ready
- ✅ Data format validated
- ⏳ Three.js viewer (Phase 2) - not started yet
- ⏳ Integration button (Phase 6) - not started yet

For now, you can test the export by running the generation scripts directly.
""")
