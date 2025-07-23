class DataManager {
    constructor() {
        this.shapes = new Map(); // className -> shapes array
        this.classVisibility = new Map();
        this.classColors = new Map();
        this.bounds = {
            min: { x: Infinity, y: Infinity, z: Infinity },
            max: { x: -Infinity, y: -Infinity, z: -Infinity }
        };
    }

    addWire(x1, y1, z, x2, y2, z2, comment, shapeClass, color) {
        // Only allow horizontal or vertical lines
        if (!(Math.abs(x1 - x2) < 1e-6 || Math.abs(y1 - y2) < 1e-6)) {
            return;
        }

        const shape = {
            type: 'Wire',
            x1, y1, z1: z,
            x2, y2, z2: z,
            comment,
            shapeClass,
            width: 5.0
        };

        this._addShape(shape, shapeClass, color);
    }

    addRect(x1, y1, z, x2, y2, z2, comment, shapeClass, color) {
        const shape = {
            type: 'Rect',
            x1, y1, z1: z,
            x2, y2, z2: z,
            comment,
            shapeClass,
            width: 0.0
        };

        this._addShape(shape, shapeClass, color);
    }

    addVia(x, y, z1, z2, comment, shapeClass, color) {
        const shape = {
            type: 'Via',
            x1: x, y1: y, z1,
            x2: x, y2: y, z2,
            comment,
            shapeClass,
            width: 5.0
        };

        this._addShape(shape, shapeClass, color);
    }

    _addShape(shape, shapeClass, color) {
        if (!this.shapes.has(shapeClass)) {
            this.shapes.set(shapeClass, []);
            this.classVisibility.set(shapeClass, true);
            this.classColors.set(shapeClass, color || { r: 1, g: 1, b: 1 });
        }

        this.shapes.get(shapeClass).push(shape);
        this._updateBounds(shape);
    }

    _updateBounds(shape) {
        const coords = [
            { x: shape.x1, y: shape.y1, z: shape.z1 },
            { x: shape.x2, y: shape.y2, z: shape.z2 }
        ];

        coords.forEach(coord => {
            this.bounds.min.x = Math.min(this.bounds.min.x, coord.x);
            this.bounds.min.y = Math.min(this.bounds.min.y, coord.y);
            this.bounds.min.z = Math.min(this.bounds.min.z, coord.z);
            this.bounds.max.x = Math.max(this.bounds.max.x, coord.x);
            this.bounds.max.y = Math.max(this.bounds.max.y, coord.y);
            this.bounds.max.z = Math.max(this.bounds.max.z, coord.z);
        });
    }

    autoScale() {
        if (this.shapes.size === 0) return;

        const dx = this.bounds.max.x - this.bounds.min.x;
        const dy = this.bounds.max.y - this.bounds.min.y;
        const dz = (this.bounds.max.z - this.bounds.min.z) * 5.0;

        if (dx < 1e-6 || dy < 1e-6 || dz < 1e-6) return;

        this.shapes.forEach(shapeList => {
            shapeList.forEach(shape => {
                shape.x1 = ((shape.x1 - this.bounds.min.x) / dx) * 100.0;
                shape.x2 = ((shape.x2 - this.bounds.min.x) / dx) * 100.0;
                shape.y1 = ((shape.y1 - this.bounds.min.y) / dy) * 100.0;
                shape.y2 = ((shape.y2 - this.bounds.min.y) / dy) * 100.0;
                shape.z1 = ((shape.z1 - this.bounds.min.z) / dz) * 100.0;
                shape.z2 = ((shape.z2 - this.bounds.min.z) / dz) * 100.0;
            });
        });

        // Update bounds after scaling
        this.bounds = {
            min: { x: 0, y: 0, z: 0 },
            max: { x: 100, y: 100, z: 100 }
        };
    }

    loadFromJSON(jsonData) {
        this.clear();
        
        if (jsonData.shapes) {
            jsonData.shapes.forEach(item => {
                const color = item.color || { r: 1, g: 1, b: 1 };
                
                switch (item.type) {
                    case 'Wire':
                        this.addWire(
                            item.x1, item.y1, item.z1,
                            item.x2, item.y2, item.z2,
                            item.comment || '',
                            item.shapeClass || 'default',
                            color
                        );
                        break;
                    case 'Rect':
                        this.addRect(
                            item.x1, item.y1, item.z1,
                            item.x2, item.y2, item.z2,
                            item.comment || '',
                            item.shapeClass || 'default',
                            color
                        );
                        break;
                    case 'Via':
                        this.addVia(
                            item.x1, item.y1, item.z1, item.z2,
                            item.comment || '',
                            item.shapeClass || 'default',
                            color
                        );
                        break;
                }
            });
        }

        this.autoScale();
    }

    clear() {
        this.shapes.clear();
        this.classVisibility.clear();
        this.classColors.clear();
        this.bounds = {
            min: { x: Infinity, y: Infinity, z: Infinity },
            max: { x: -Infinity, y: -Infinity, z: -Infinity }
        };
    }

    getShapesByClass(className) {
        return this.shapes.get(className) || [];
    }

    isClassVisible(className) {
        return this.classVisibility.get(className) !== false;
    }

    setClassVisibility(className, visible) {
        this.classVisibility.set(className, visible);
    }

    getClassColor(className) {
        return this.classColors.get(className) || { r: 1, g: 1, b: 1 };
    }

    setClassColor(className, color) {
        this.classColors.set(className, color);
    }

    getAllClasses() {
        return Array.from(this.shapes.keys());
    }

    getBounds() {
        return this.bounds;
    }
}
