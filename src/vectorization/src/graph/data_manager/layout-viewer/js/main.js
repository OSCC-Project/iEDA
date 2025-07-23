// Main application initialization
class App {
    constructor() {
        this.init();
    }

    init() {
        const canvas = document.getElementById('canvas');
        this.sceneManager = new SceneManager(canvas);
        this.controlsManager = new ControlsManager(this.sceneManager);

        // Add some sample data for demonstration
        this.addSampleData();

        console.log('3D Layout Viewer initialized');
        console.log('Server running at http://localhost:19999');
    }

    addSampleData() {
        // Add some sample shapes to demonstrate functionality
        const colors = [
            { r: 1, g: 0, b: 0 },     // Red
            { r: 0, g: 1, b: 0 },     // Green
            { r: 0, g: 0, b: 1 },     // Blue
            { r: 1, g: 1, b: 0 },     // Yellow
            { r: 1, g: 0, b: 1 },     // Magenta
        ];

        // Add sample wires
        for (let i = 0; i < 5; i++) {
            this.sceneManager.addWire(
                i * 20, 0, 0,
                i * 20, 50, 0,
                `Sample wire ${i + 1}`,
                `Wire_Class_${i + 1}`,
                colors[i % colors.length]
            );
        }

        // Add sample rectangles
        for (let i = 0; i < 3; i++) {
            this.sceneManager.addRect(
                i * 30, 20, 5,
                i * 30 + 15, 35, 5,
                `Sample rect ${i + 1}`,
                `Rect_Class_${i + 1}`,
                colors[i % colors.length]
            );
        }

        // Add sample vias
        for (let i = 0; i < 4; i++) {
            this.sceneManager.addVia(
                i * 25, 40, 0, 20,
                `Sample via ${i + 1}`,
                `Via_Class_${i + 1}`,
                colors[i % colors.length]
            );
        }

        this.sceneManager.dataManager.autoScale();
        this.sceneManager.resetView();
        this.controlsManager.updateClassTable();
        this.controlsManager.updateGroupsDisplay();
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new App();
});

// Global error handling
window.addEventListener('error', (e) => {
    console.error('Application error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});
