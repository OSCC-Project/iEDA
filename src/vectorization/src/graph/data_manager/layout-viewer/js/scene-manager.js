class SceneManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 10000);
        this.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        this.dataManager = new DataManager();
        
        this.meshGroups = new Map(); // className -> THREE.Group
        this.axesHelper = null;
        this.axesVisible = false;
        
        // Custom mouse controls
        this.isMouseDown = false;
        this.lastMousePos = { x: 0, y: 0 };
        this.isRotationMode = false;
        this.panSpeed = 0.5;
        this.rotationSpeed = 1.0;
        
        // Keyboard state
        this.keys = {
            ctrl: false
        };
        
        // Pan state
        this.panOffset = { x: 0, y: 0 };
        
        this.setupScene();
        this.setupEventListeners();
        this.setupCustomControls();
    }

    setupScene() {
        this.scene.background = new THREE.Color(0x262626);
        
        // Setup renderer
        this.renderer.setSize(this.canvas.parentElement.clientWidth, this.canvas.parentElement.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Setup camera with initial view: XY plane at bottom, Z-axis pointing up
        this.setupInitialCameraPosition();

        // Setup lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
        directionalLight.position.set(100, 100, 50);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);

        // Create a group for all objects that we want to rotate/pan
        this.objectGroup = new THREE.Group();
        this.scene.add(this.objectGroup);

        this.animate();
    }

    setupInitialCameraPosition() {
        // Position camera to view XY plane from above at an angle
        // XY plane is at the bottom (Z=0), positive Z points up
        this.camera.position.set(50, 50, 50);
        this.camera.lookAt(0, 0, 0);
        
        // Set camera up vector to ensure Z is up
        this.camera.up.set(0, 0, 1);
        this.camera.updateProjectionMatrix();
    }

    calculateOptimalView(bounds) {
        // Calculate data dimensions
        const sizeX = bounds.max.x - bounds.min.x;
        const sizeY = bounds.max.y - bounds.min.y;
        const sizeZ = bounds.max.z - bounds.min.z;
        
        // Find the maximum dimension to determine overall scale
        const maxSize = Math.max(sizeX, sizeY, sizeZ);
        
        // Calculate center point
        const center = {
            x: (bounds.min.x + bounds.max.x) / 2,
            y: (bounds.min.y + bounds.max.y) / 2,
            z: (bounds.min.z + bounds.max.z) / 2
        };
        
        // Calculate camera distance to fit all data
        // Use field of view to determine appropriate distance
        const fov = this.camera.fov * Math.PI / 180; // Convert to radians
        const aspect = this.camera.aspect;
        
        // Calculate distance needed to fit the data with some padding
        const padding = 1.5; // Add 50% padding around the data
        const distance = (maxSize * padding) / (2 * Math.tan(fov / 2));
        
        // Position camera at optimal distance with Z-up orientation
        // Place camera at an angle to show 3D structure clearly
        const cameraOffset = distance * 0.8;
        
        return {
            center,
            distance,
            cameraOffset,
            maxSize,
            bounds
        };
    }

    updateAxesHelper(bounds) {
        // Remove existing axes
        if (this.axesHelper) {
            this.objectGroup.remove(this.axesHelper);
            this.axesHelper = null;
        }
        
        // Create new axes that encompass all data
        if (this.axesVisible) {
            const sizeX = bounds.max.x - bounds.min.x;
            const sizeY = bounds.max.y - bounds.min.y;
            const sizeZ = bounds.max.z - bounds.min.z;
            
            // Make axes slightly larger than data bounds
            const axisLength = Math.max(sizeX, sizeY, sizeZ) * 1.2;
            
            // Create custom axes helper positioned at data origin
            this.axesHelper = new THREE.Group();
            
            // X-axis (red)
            const xGeometry = new THREE.BufferGeometry();
            xGeometry.setAttribute('position', new THREE.BufferAttribute(
                new Float32Array([
                    bounds.min.x - axisLength * 0.1, 0, 0,
                    bounds.max.x + axisLength * 0.1, 0, 0
                ]), 3));
            const xMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 3 });
            const xLine = new THREE.Line(xGeometry, xMaterial);
            this.axesHelper.add(xLine);
            
            // Y-axis (green)
            const yGeometry = new THREE.BufferGeometry();
            yGeometry.setAttribute('position', new THREE.BufferAttribute(
                new Float32Array([
                    0, bounds.min.y - axisLength * 0.1, 0,
                    0, bounds.max.y + axisLength * 0.1, 0
                ]), 3));
            const yMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 3 });
            const yLine = new THREE.Line(yGeometry, yMaterial);
            this.axesHelper.add(yLine);
            
            // Z-axis (blue)
            const zGeometry = new THREE.BufferGeometry();
            zGeometry.setAttribute('position', new THREE.BufferAttribute(
                new Float32Array([
                    0, 0, bounds.min.z - axisLength * 0.1,
                    0, 0, bounds.max.z + axisLength * 0.1
                ]), 3));
            const zMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff, linewidth: 3 });
            const zLine = new THREE.Line(zGeometry, zMaterial);
            this.axesHelper.add(zLine);
            
            this.objectGroup.add(this.axesHelper);
        }
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Keyboard events for Ctrl key detection
        window.addEventListener('keydown', (event) => {
            if (event.key === 'Control' || event.ctrlKey) {
                this.keys.ctrl = true;
            }
        });
        
        window.addEventListener('keyup', (event) => {
            if (event.key === 'Control' || !event.ctrlKey) {
                this.keys.ctrl = false;
            }
        });
        
        // Handle focus loss to reset key states
        window.addEventListener('blur', () => {
            this.keys.ctrl = false;
        });
        
        // Mouse events for tooltip
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.canvas.addEventListener('mousemove', (event) => this.onMouseMove(event));
    }

    setupCustomControls() {
        this.canvas.addEventListener('mousedown', (event) => this.onMouseDown(event));
        this.canvas.addEventListener('mousemove', (event) => this.onMouseMoveCustom(event));
        this.canvas.addEventListener('mouseup', (event) => this.onMouseUp(event));
        this.canvas.addEventListener('mouseleave', (event) => this.onMouseUp(event)); // Handle mouse leaving canvas
        this.canvas.addEventListener('wheel', (event) => this.onMouseWheel(event));
        
        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    }

    onMouseDown(event) {
        // Skip if clicking on control panel
        if (event.clientX < 300) return;
        
        this.isMouseDown = true;
        const rect = this.canvas.getBoundingClientRect();
        this.lastMousePos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        
        this.canvas.style.cursor = this.isRotationMode ? 'grabbing' : 'move';
    }

    onMouseMoveCustom(event) {
        // Skip if clicking on control panel
        if (event.clientX < 300) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentPos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        
        if (this.isMouseDown) {
            const deltaX = currentPos.x - this.lastMousePos.x;
            const deltaY = currentPos.y - this.lastMousePos.y;
            
            if (this.isRotationMode) {
                // Rotation mode - check for Ctrl key
                if (this.keys.ctrl) {
                    this.rotateSceneWorldZAxis(deltaX);
                } else {
                    this.rotateScene(deltaX, deltaY);
                }
            } else {
                // Pan mode - reversed direction
                this.panScene(deltaX, deltaY);
            }
        }
        
        this.lastMousePos = currentPos;
    }

    onMouseUp(event) {
        if (!this.isMouseDown) return;
        
        this.isMouseDown = false;
        this.canvas.style.cursor = 'default';
    }

    onMouseWheel(event) {
        // Skip if over control panel
        if (event.clientX < 300) return;
        
        event.preventDefault();
        
        const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9;
        this.camera.position.multiplyScalar(zoomFactor);
    }

    rotateScene(deltaX, deltaY) {
        // More intuitive rotation that follows mouse movement
        // The scene rotates as if you're grabbing and turning a physical object
        
        // Get the camera's current view direction and up vector
        const cameraDirection = new THREE.Vector3();
        this.camera.getWorldDirection(cameraDirection);
        
        // Calculate rotation axes relative to camera view
        const cameraUp = this.camera.up.clone();
        const cameraRight = new THREE.Vector3().crossVectors(cameraUp, cameraDirection).normalize();
        
        // Horizontal mouse movement (deltaX) rotates around the world Z-axis (since Z is up)
        // This makes left/right mouse movement rotate the object left/right as expected
        const zRotation = -deltaX * 0.01 * this.rotationSpeed;
        this.objectGroup.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), zRotation);
        
        // Vertical mouse movement (deltaY) rotates around the camera's right vector
        // This makes up/down mouse movement tilt the object up/down as expected
        const xRotation = -deltaY * 0.01 * this.rotationSpeed;
        
        // Transform the camera's right vector to world space for consistent rotation
        const worldRight = cameraRight.clone();
        this.objectGroup.rotateOnWorldAxis(worldRight, xRotation);
    }

    rotateSceneWorldZAxis(deltaX) {
        // Z-axis only rotation around the world coordinate system Z-axis
        // This rotates around the absolute Z-axis of the coordinate system,
        // not the viewport/camera Z-axis. Positive deltaX rotates counterclockwise
        // when viewed from above (looking down the positive Z-axis).
        const zRotation = -deltaX * 0.01 * this.rotationSpeed;
        
        // Use the world Z-axis vector explicitly to ensure strict Z-axis rotation
        const worldZAxis = new THREE.Vector3(0, 0, 1);
        this.objectGroup.rotateOnWorldAxis(worldZAxis, zRotation);
    }

    panScene(deltaX, deltaY) {
        // Convert screen space movement to world space
        // Reversed direction: positive deltaX moves scene left, positive deltaY moves scene down
        
        // Get camera's right and up vectors
        const cameraMatrix = this.camera.matrixWorld.clone();
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        
        right.setFromMatrixColumn(cameraMatrix, 0); // X column
        up.setFromMatrixColumn(cameraMatrix, 1);    // Y column
        
        // Calculate pan distance based on camera distance
        const distance = this.camera.position.length();
        const panScale = distance * 0.001 * this.panSpeed;
        
        // Apply panning with reversed direction
        const panVector = new THREE.Vector3();
        panVector.addScaledVector(right, deltaX * panScale);  // Reversed: removed minus sign
        panVector.addScaledVector(up, -deltaY * panScale);   // Reversed: changed plus to minus
        
        this.objectGroup.position.add(panVector);
    }

    setRotationMode(enabled) {
        this.isRotationMode = enabled;
        this.canvas.style.cursor = 'default';
    }

    onWindowResize() {
        const container = this.canvas.parentElement;
        const width = container.clientWidth;
        const height = container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    onMouseMove(event) {
        // This is for tooltip functionality only
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const tooltip = document.getElementById('tooltip');
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        if (intersects.length > 0) {
            const intersect = intersects[0];
            const userData = intersect.object.userData;
            
            // Only show tooltip if the object has a comment AND its class is visible
            if (userData.comment && userData.shapeClass && this.dataManager.isClassVisible(userData.shapeClass)) {
                tooltip.style.display = 'block';
                tooltip.style.left = event.clientX + 10 + 'px';
                tooltip.style.top = event.clientY + 10 + 'px';
                tooltip.textContent = userData.comment;
            } else {
                tooltip.style.display = 'none';
            }
        } else {
            tooltip.style.display = 'none';
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    addWire(x1, y1, z1, x2, y2, z2, comment, shapeClass, color) {
        this.dataManager.addWire(x1, y1, z1, x2, y2, z2, comment, shapeClass, color);
        this.createWireMesh(x1, y1, z1, x2, y2, z2, comment, shapeClass, color);
    }

    addRect(x1, y1, z1, x2, y2, z2, comment, shapeClass, color) {
        this.dataManager.addRect(x1, y1, z1, x2, y2, z2, comment, shapeClass, color);
        this.createRectMesh(x1, y1, z1, x2, y2, z2, comment, shapeClass, color);
    }

    addVia(x, y, z1, z2, comment, shapeClass, color) {
        this.dataManager.addVia(x, y, z1, z2, comment, shapeClass, color);
        this.createViaMesh(x, y, z1, z2, comment, shapeClass, color);
    }

    createWireMesh(x1, y1, z1, x2, y2, z2, comment, shapeClass, color) {
        // Calculate wire direction and length
        const direction = new THREE.Vector3(x2 - x1, y2 - y1, z2 - z1);
        const length = direction.length();
        direction.normalize();

        // Create a cylindrical geometry for the wire with visible thickness
        const wireRadius = Math.max(0.1, length * 0.002); // Dynamic radius based on wire length
        const geometry = new THREE.CylinderGeometry(wireRadius, wireRadius, length, 8);

        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(color.r, color.g, color.b),
            transparent: color.a !== undefined,
            opacity: color.a !== undefined ? color.a : 1.0
        });

        const mesh = new THREE.Mesh(geometry, material);
        
        // Position the mesh at the midpoint of the wire
        const midpoint = new THREE.Vector3(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            (z1 + z2) / 2
        );
        mesh.position.copy(midpoint);

        // Align the cylinder with the wire direction
        const up = new THREE.Vector3(0, 1, 0);
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(up, direction);
        mesh.setRotationFromQuaternion(quaternion);

        mesh.userData = { comment, shapeClass, type: 'Wire' };

        this.addToGroup(mesh, shapeClass);
    }

    createRectMesh(x1, y1, z1, x2, y2, z2, comment, shapeClass, color) {
        const width = Math.abs(x2 - x1);
        const height = Math.abs(y2 - y1);
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;

        const geometry = new THREE.PlaneGeometry(width, height);
        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(color.r, color.g, color.b),
            side: THREE.DoubleSide,
            transparent: true,
            opacity: color.a !== undefined ? color.a : 0.85
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(centerX, centerY, z1);
        mesh.userData = { comment, shapeClass, type: 'Rect' };

        this.addToGroup(mesh, shapeClass);
    }

    createViaMesh(x, y, z1, z2, comment, shapeClass, color) {
        // Calculate via direction and length
        const length = Math.abs(z2 - z1);
        const centerZ = (z1 + z2) / 2;

        // Create a cylindrical geometry for the via with half the width of wire
        // Wire radius is: Math.max(0.1, length * 0.002)
        // Via radius is half of that
        const viaRadius = Math.max(0.05, length * 0.001); // Half of wire radius
        const geometry = new THREE.CylinderGeometry(viaRadius, viaRadius, length, 8);

        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(color.r, color.g, color.b),
            transparent: color.a !== undefined,
            opacity: color.a !== undefined ? color.a : 1.0
        });

        const mesh = new THREE.Mesh(geometry, material);
        
        // Position the mesh at the center point
        mesh.position.set(x, y, centerZ);

        // Rotate to align with Z-axis (cylinder default is Y-axis)
        mesh.rotateX(Math.PI / 2);

        mesh.userData = { comment, shapeClass, type: 'Via' };

        this.addToGroup(mesh, shapeClass);
    }

    addToGroup(mesh, shapeClass) {
        if (!this.meshGroups.has(shapeClass)) {
            const group = new THREE.Group();
            group.name = shapeClass;
            this.meshGroups.set(shapeClass, group);
            this.objectGroup.add(group); // Add to objectGroup instead of scene
        }

        this.meshGroups.get(shapeClass).add(mesh);
    }

    setClassVisibility(className, visible) {
        this.dataManager.setClassVisibility(className, visible);
        const group = this.meshGroups.get(className);
        if (group) {
            group.visible = visible;
        }
    }

    setClassColor(className, color) {
        this.dataManager.setClassColor(className, color);
        const group = this.meshGroups.get(className);
        if (group) {
            group.children.forEach(child => {
                if (child.material) {
                    child.material.color.setRGB(color.r, color.g, color.b);
                    
                    // Handle alpha transparency
                    if (color.a !== undefined) {
                        child.material.transparent = true;
                        child.material.opacity = color.a;
                    } else {
                        child.material.transparent = false;
                        child.material.opacity = 1.0;
                    }
                    
                    // Update material to reflect changes
                    child.material.needsUpdate = true;
                }
            });
        }
    }

    showAxes(show = true) {
        this.axesVisible = show;
        
        // Update axes based on current data bounds
        const bounds = this.dataManager.getBounds();
        if (bounds.min.x !== Infinity) {
            this.updateAxesHelper(bounds);
        } else if (!show && this.axesHelper) {
            // Remove axes if no data and hiding
            this.objectGroup.remove(this.axesHelper);
            this.axesHelper = null;
        }
    }

    toggleAxes() {
        this.showAxes(!this.axesVisible);
    }

    resetView() {
        // Reset object group transform
        this.objectGroup.position.set(0, 0, 0);
        this.objectGroup.rotation.set(0, 0, 0);
        
        // Reset camera to optimal position based on data bounds
        const bounds = this.dataManager.getBounds();
        if (bounds.min.x === Infinity) {
            // No data, use default position with Z-up
            this.setupInitialCameraPosition();
        } else {
            const viewInfo = this.calculateOptimalView(bounds);
            
            // Position camera at optimal distance with Z-up orientation
            this.camera.position.set(
                viewInfo.center.x + viewInfo.cameraOffset,
                viewInfo.center.y + viewInfo.cameraOffset,
                viewInfo.center.z + viewInfo.cameraOffset
            );
            
            this.camera.lookAt(viewInfo.center.x, viewInfo.center.y, viewInfo.center.z);
            
            // Ensure Z is up
            this.camera.up.set(0, 0, 1);
            this.camera.updateProjectionMatrix();
            
            // Update axes to match data bounds
            if (this.axesVisible) {
                this.updateAxesHelper(bounds);
            }
        }
    }

    clearScene() {
        // Remove all mesh groups
        this.meshGroups.forEach(group => {
            this.objectGroup.remove(group);
        });
        this.meshGroups.clear();

        // Remove axes
        if (this.axesHelper) {
            this.objectGroup.remove(this.axesHelper);
            this.axesHelper = null;
        }

        // Clear data
        this.dataManager.clear();
    }

    loadFromJSON(jsonData) {
        this.clearScene();
        this.dataManager.loadFromJSON(jsonData);
        this.rebuildScene();
        this.resetView();
    }

    rebuildScene() {
        this.dataManager.shapes.forEach((shapes, className) => {
            const color = this.dataManager.getClassColor(className);
            shapes.forEach(shape => {
                switch (shape.type) {
                    case 'Wire':
                        this.createWireMesh(
                            shape.x1, shape.y1, shape.z1,
                            shape.x2, shape.y2, shape.z2,
                            shape.comment, shape.shapeClass, color
                        );
                        break;
                    case 'Rect':
                        this.createRectMesh(
                            shape.x1, shape.y1, shape.z1,
                            shape.x2, shape.y2, shape.z2,
                            shape.comment, shape.shapeClass, color
                        );
                        break;
                    case 'Via':
                        this.createViaMesh(
                            shape.x1, shape.y1, shape.z1, shape.z2,
                            shape.comment, shape.shapeClass, color
                        );
                        break;
                }
            });
        });
    }

    getDataManager() {
        return this.dataManager;
    }
}
