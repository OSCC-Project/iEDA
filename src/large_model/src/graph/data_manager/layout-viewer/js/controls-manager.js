class ControlsManager {
    constructor(sceneManager) {
        this.sceneManager = sceneManager;
        this.dataManager = sceneManager.getDataManager();
        
        this.searchFilter = '';
        this.isRegexFilter = false;
        this.filteredClasses = new Set();
        this.groups = new Map(); // groupName -> Set of classNames
        this.classGroups = new Map(); // className -> Set of groupNames (allow multiple groups)
        
        this.currentColorCallback = null;
        this.currentGroupCallback = null;
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // View controls
        document.getElementById('rotation-mode').addEventListener('click', (e) => {
            const btn = e.target;
            const isRotation = !btn.classList.contains('active');
            btn.classList.toggle('active');
            btn.textContent = isRotation ? 'Rotation Mode (On)' : 'Rotation Mode (Off)';
            btn.title = isRotation ? 'Hold Ctrl for world Z-axis only rotation' : 'Click to enable rotation mode';
            
            // Set rotation mode in scene manager
            this.sceneManager.setRotationMode(isRotation);
        });

        document.getElementById('reset-view').addEventListener('click', () => {
            this.sceneManager.resetView();
        });

        document.getElementById('show-axes').addEventListener('click', () => {
            this.sceneManager.toggleAxes();
        });

        // File import
        document.getElementById('import-btn').addEventListener('click', () => {
            document.getElementById('file-input').click();
        });

        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileImport(e.target.files[0]);
        });

        // Search and filter
        document.getElementById('search-input').addEventListener('input', (e) => {
            this.handleSearch(e.target.value);
        });

        document.getElementById('regex-filter').addEventListener('change', (e) => {
            this.isRegexFilter = e.target.checked;
            this.handleSearch(this.searchFilter); // Re-filter with current search text
        });

        document.getElementById('hide-all').addEventListener('change', (e) => {
            if (e.target.checked) {
                document.getElementById('show-all').checked = false;
                this.setAllClassesVisibility(false);
            }
        });

        document.getElementById('show-all').addEventListener('change', (e) => {
            if (e.target.checked) {
                document.getElementById('hide-all').checked = false;
                this.setAllClassesVisibility(true);
            }
        });

        document.getElementById('unified-color').addEventListener('click', () => {
            this.handleUnifiedColor();
        });

        document.getElementById('group-filtered').addEventListener('click', () => {
            this.handleGroupFiltered();
        });

        // Color picker modal
        this.setupColorPickerModal();
        
        // Group modal
        this.setupGroupModal();
    }

    setupColorPickerModal() {
        const modal = document.getElementById('color-picker-modal');
        const colorPicker = document.getElementById('color-picker');
        const colorPreview = document.getElementById('color-preview');
        const redInput = document.getElementById('red-input');
        const greenInput = document.getElementById('green-input');
        const blueInput = document.getElementById('blue-input');
        const applyBtn = document.getElementById('apply-color');
        const cancelBtn = document.getElementById('cancel-color');
        const closeBtn = modal.querySelector('.close-btn');

        // Close modal events
        [closeBtn, cancelBtn].forEach(btn => {
            btn.addEventListener('click', () => {
                modal.style.display = 'none';
                this.currentColorCallback = null;
            });
        });

        // Click outside to close
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
                this.currentColorCallback = null;
            }
        });

        // Color picker change
        colorPicker.addEventListener('input', (e) => {
            const hex = e.target.value;
            const rgb = this.hexToRgb(hex);
            redInput.value = rgb.r;
            greenInput.value = rgb.g;
            blueInput.value = rgb.b;
            colorPreview.style.backgroundColor = hex;
        });

        // RGB input changes
        [redInput, greenInput, blueInput].forEach(input => {
            input.addEventListener('input', () => {
                const r = Math.max(0, Math.min(255, parseInt(redInput.value) || 0));
                const g = Math.max(0, Math.min(255, parseInt(greenInput.value) || 0));
                const b = Math.max(0, Math.min(255, parseInt(blueInput.value) || 0));
                
                redInput.value = r;
                greenInput.value = g;
                blueInput.value = b;
                
                const hex = this.rgbToHex(r/255, g/255, b/255);
                colorPicker.value = hex;
                colorPreview.style.backgroundColor = hex;
            });
        });

        // Apply color
        applyBtn.addEventListener('click', () => {
            if (this.currentColorCallback) {
                const r = parseInt(redInput.value) / 255;
                const g = parseInt(greenInput.value) / 255;
                const b = parseInt(blueInput.value) / 255;
                this.currentColorCallback({ r, g, b });
            }
            modal.style.display = 'none';
            this.currentColorCallback = null;
        });
    }

    setupGroupModal() {
        const modal = document.getElementById('group-modal');
        const groupSelect = document.getElementById('group-select');
        const newGroupInput = document.getElementById('new-group-name');
        const assignBtn = document.getElementById('assign-group');
        const cancelBtn = document.getElementById('cancel-group');
        const closeBtn = modal.querySelector('.close-btn');

        // Close modal events
        [closeBtn, cancelBtn].forEach(btn => {
            btn.addEventListener('click', () => {
                modal.style.display = 'none';
                this.currentGroupCallback = null;
            });
        });

        // Click outside to close
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
                this.currentGroupCallback = null;
            }
        });

        // Clear new group input when selecting existing group
        groupSelect.addEventListener('change', () => {
            if (groupSelect.value) {
                newGroupInput.value = '';
            }
        });

        // Clear group selection when typing new group name
        newGroupInput.addEventListener('input', () => {
            if (newGroupInput.value.trim()) {
                groupSelect.value = '';
            }
        });

        // Assign to group
        assignBtn.addEventListener('click', () => {
            const selectedGroup = groupSelect.value;
            const newGroupName = newGroupInput.value.trim();
            
            if (selectedGroup || newGroupName) {
                const groupName = newGroupName || selectedGroup;
                if (this.currentGroupCallback) {
                    this.currentGroupCallback(groupName);
                }
            }
            
            modal.style.display = 'none';
            this.currentGroupCallback = null;
        });
    }

    showColorPicker(initialColor, callback) {
        const modal = document.getElementById('color-picker-modal');
        const colorPicker = document.getElementById('color-picker');
        const colorPreview = document.getElementById('color-preview');
        const redInput = document.getElementById('red-input');
        const greenInput = document.getElementById('green-input');
        const blueInput = document.getElementById('blue-input');

        // Set initial values
        const hex = this.rgbToHex(initialColor.r, initialColor.g, initialColor.b);
        colorPicker.value = hex;
        colorPreview.style.backgroundColor = hex;
        redInput.value = Math.round(initialColor.r * 255);
        greenInput.value = Math.round(initialColor.g * 255);
        blueInput.value = Math.round(initialColor.b * 255);

        this.currentColorCallback = callback;
        modal.style.display = 'block';
    }

    showGroupModal(callback) {
        const modal = document.getElementById('group-modal');
        const groupSelect = document.getElementById('group-select');
        const newGroupInput = document.getElementById('new-group-name');

        // Populate group options
        groupSelect.innerHTML = '<option value="">-- Select Group --</option>';
        for (const groupName of this.groups.keys()) {
            const option = document.createElement('option');
            option.value = groupName;
            option.textContent = groupName;
            groupSelect.appendChild(option);
        }

        // Clear inputs
        groupSelect.value = '';
        newGroupInput.value = '';

        this.currentGroupCallback = callback;
        modal.style.display = 'block';
    }

    async handleFileImport(file) {
        if (!file) return;

        const statusDiv = document.getElementById('file-status');
        
        try {
            statusDiv.className = 'status-info';
            statusDiv.textContent = 'Uploading file...';

            const formData = new FormData();
            formData.append('jsonFile', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                statusDiv.className = 'status-success';
                statusDiv.textContent = `Successfully loaded ${file.name}`;
                
                this.sceneManager.loadFromJSON(result.data);
                this.updateClassTable();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('File import error:', error);
            statusDiv.className = 'status-error';
            statusDiv.textContent = `Error: ${error.message}`;
        }

        // Reset file input
        document.getElementById('file-input').value = '';
    }

    handleSearch(searchText) {
        this.searchFilter = searchText;
        this.updateClassTable();
    }

    setAllClassesVisibility(visible) {
        this.dataManager.getAllClasses().forEach(className => {
            this.dataManager.setClassVisibility(className, visible);
            this.sceneManager.setClassVisibility(className, visible);
        });
        this.updateClassTable();
    }

    handleUnifiedColor() {
        // Get currently filtered classes only
        const filteredClasses = this.getFilteredClasses();
        
        this.showColorPicker({ r: 1, g: 0, b: 0 }, (color) => {
            filteredClasses.forEach(className => {
                if (this.dataManager.isClassVisible(className)) {
                    this.dataManager.setClassColor(className, color);
                    this.sceneManager.setClassColor(className, color);
                }
            });
            this.updateClassTable();
        });
    }

    // Helper method to get currently filtered classes
    getFilteredClasses() {
        const allClasses = this.dataManager.getAllClasses();
        
        if (this.searchFilter === '') {
            return allClasses;
        } else if (this.isRegexFilter) {
            try {
                const regex = new RegExp(this.searchFilter, 'i');
                return allClasses.filter(className => regex.test(className));
            } catch (e) {
                // Invalid regex, fall back to plain text search
                const lowerFilter = this.searchFilter.toLowerCase();
                return allClasses.filter(className => 
                    className.toLowerCase().includes(lowerFilter)
                );
            }
        } else {
            const lowerFilter = this.searchFilter.toLowerCase();
            return allClasses.filter(className => 
                className.toLowerCase().includes(lowerFilter)
            );
        }
    }

    handleGroupFiltered() {
        const allClasses = this.dataManager.getAllClasses();
        
        let filteredClasses;
        if (this.searchFilter === '') {
            filteredClasses = allClasses;
        } else if (this.isRegexFilter) {
            try {
                const regex = new RegExp(this.searchFilter, 'i');
                filteredClasses = allClasses.filter(className => regex.test(className));
            } catch (e) {
                // Invalid regex, fall back to plain text search
                const lowerFilter = this.searchFilter.toLowerCase();
                filteredClasses = allClasses.filter(className => 
                    className.toLowerCase().includes(lowerFilter)
                );
            }
        } else {
            const lowerFilter = this.searchFilter.toLowerCase();
            filteredClasses = allClasses.filter(className => 
                className.toLowerCase().includes(lowerFilter)
            );
        }

        if (filteredClasses.length === 0) {
            alert('No classes match the current filter.');
            return;
        }

        this.showGroupModal((groupName) => {
            this.assignClassesToGroup(filteredClasses, groupName);
        });
    }

    assignClassesToGroup(classNames, groupName) {
        // Add classes to new group (allow multiple group membership)
        if (!this.groups.has(groupName)) {
            this.groups.set(groupName, new Set());
        }
        
        classNames.forEach(className => {
            // Add class to the group
            this.groups.get(groupName).add(className);
            
            // Add group to class's group set
            if (!this.classGroups.has(className)) {
                this.classGroups.set(className, new Set());
            }
            this.classGroups.get(className).add(groupName);
        });

        this.updateClassTable();
        this.updateGroupsDisplay();
        
        console.log(`Added ${classNames.length} classes to group "${groupName}". Classes can belong to multiple groups simultaneously.`);
    }

    assignClassToGroup(className, groupName) {
        this.assignClassesToGroup([className], groupName);
    }

    updateClassTable() {
        const classList = document.getElementById('class-list');
        classList.innerHTML = '';

        const allClasses = this.dataManager.getAllClasses();
        
        let filteredClasses;
        if (this.searchFilter === '') {
            filteredClasses = allClasses;
        } else if (this.isRegexFilter) {
            try {
                const regex = new RegExp(this.searchFilter, 'i');
                filteredClasses = allClasses.filter(className => regex.test(className));
            } catch (e) {
                // Invalid regex, fall back to plain text search
                const lowerFilter = this.searchFilter.toLowerCase();
                filteredClasses = allClasses.filter(className => 
                    className.toLowerCase().includes(lowerFilter)
                );
            }
        } else {
            const lowerFilter = this.searchFilter.toLowerCase();
            filteredClasses = allClasses.filter(className => 
                className.toLowerCase().includes(lowerFilter)
            );
        }

        filteredClasses.forEach(className => {
            const row = this.createClassRow(className);
            classList.appendChild(row);
        });
    }

    createClassRow(className) {
        const row = document.createElement('div');
        row.className = 'table-row';

        // Class name
        const nameCell = document.createElement('span');
        nameCell.textContent = className;
        nameCell.title = className;

        // Visibility checkbox
        const visibleCell = document.createElement('input');
        visibleCell.type = 'checkbox';
        visibleCell.checked = this.dataManager.isClassVisible(className);
        visibleCell.addEventListener('change', (e) => {
            this.dataManager.setClassVisibility(className, e.target.checked);
            this.sceneManager.setClassVisibility(className, e.target.checked);
        });

        // Color picker
        const color = this.dataManager.getClassColor(className);
        const colorCell = document.createElement('div');
        colorCell.className = 'color-cell';
        colorCell.style.backgroundColor = this.rgbToHex(color.r, color.g, color.b);
        colorCell.addEventListener('click', () => {
            this.showColorPicker(color, (newColor) => {
                this.dataManager.setClassColor(className, newColor);
                this.sceneManager.setClassColor(className, newColor);
                colorCell.style.backgroundColor = this.rgbToHex(newColor.r, newColor.g, newColor.b);
            });
        });

        // Group button
        const groupBtn = document.createElement('button');
        groupBtn.className = 'group-btn';
        groupBtn.textContent = 'Group';
        groupBtn.addEventListener('click', () => {
            this.showGroupModal((groupName) => {
                this.assignClassToGroup(className, groupName);
            });
        });

        row.appendChild(nameCell);
        row.appendChild(visibleCell);
        row.appendChild(colorCell);
        row.appendChild(groupBtn);

        return row;
    }

    updateGroupsDisplay() {
        const groupsList = document.getElementById('groups-list');
        groupsList.innerHTML = '';

        for (const [groupName, classSet] of this.groups.entries()) {
            if (classSet.size === 0) continue;

            const groupItem = this.createGroupItem(groupName, classSet);
            groupsList.appendChild(groupItem);
        }
    }

    createGroupItem(groupName, classSet) {
        const groupItem = document.createElement('div');
        groupItem.className = 'group-item';

        // Group header
        const groupHeader = document.createElement('div');
        groupHeader.className = 'group-header';
        
        const groupNameSpan = document.createElement('span');
        groupNameSpan.className = 'group-name';
        groupNameSpan.textContent = `${groupName} (${classSet.size})`;
        
        // Group controls container
        const groupControls = document.createElement('div');
        groupControls.className = 'group-controls';
        
        // Group visibility toggle
        const groupVisibleBtn = document.createElement('button');
        groupVisibleBtn.className = 'group-control-btn';
        groupVisibleBtn.textContent = 'Visible';
        groupVisibleBtn.title = 'Toggle visibility for all items in this group';
        groupVisibleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleGroupVisibility(groupName, classSet);
        });
        
        // Group color button
        const groupColorBtn = document.createElement('button');
        groupColorBtn.className = 'group-control-btn group-color-btn';
        groupColorBtn.textContent = 'Color';
        groupColorBtn.title = 'Set color for all items in this group';
        groupColorBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.setGroupColor(groupName, classSet);
        });
        
        // Group blur button
        const groupBlurBtn = document.createElement('button');
        groupBlurBtn.className = 'group-control-btn group-blur-btn';
        groupBlurBtn.textContent = 'Blur';
        groupBlurBtn.title = 'Toggle transparency for all items in this group';
        groupBlurBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleGroupBlur(groupName, classSet);
        });
        
        // Group dissolve button
        const groupDissolveBtn = document.createElement('button');
        groupDissolveBtn.className = 'group-control-btn group-dissolve-btn';
        groupDissolveBtn.textContent = 'Dissolve';
        groupDissolveBtn.title = 'Delete this group (classes remain visible)';
        groupDissolveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.dissolveGroup(groupName);
        });
        
        const toggleSpan = document.createElement('span');
        toggleSpan.className = 'group-toggle';
        toggleSpan.textContent = '▼';

        groupControls.appendChild(groupVisibleBtn);
        groupControls.appendChild(groupColorBtn);
        groupControls.appendChild(groupBlurBtn);
        groupControls.appendChild(groupDissolveBtn);
        
        groupHeader.appendChild(groupNameSpan);
        groupHeader.appendChild(groupControls);
        groupHeader.appendChild(toggleSpan);

        // Group content
        const groupContent = document.createElement('div');
        groupContent.className = 'group-content';

        for (const className of classSet) {
            const itemRow = this.createGroupItemRow(className);
            groupContent.appendChild(itemRow);
        }

        // Toggle functionality
        groupHeader.addEventListener('click', (e) => {
            // Don't toggle if clicking on control buttons
            if (e.target.classList.contains('group-control-btn')) {
                return;
            }
            const isExpanded = groupContent.classList.toggle('expanded');
            toggleSpan.textContent = isExpanded ? '▲' : '▼';
        });

        groupItem.appendChild(groupHeader);
        groupItem.appendChild(groupContent);

        return groupItem;
    }

    createGroupItemRow(className) {
        const row = document.createElement('div');
        row.className = 'group-item-row';

        // Class name
        const nameSpan = document.createElement('span');
        nameSpan.textContent = className;
        nameSpan.title = className;

        // Visibility checkbox
        const visibleInput = document.createElement('input');
        visibleInput.type = 'checkbox';
        visibleInput.checked = this.dataManager.isClassVisible(className);
        visibleInput.addEventListener('change', (e) => {
            this.dataManager.setClassVisibility(className, e.target.checked);
            this.sceneManager.setClassVisibility(className, e.target.checked);
            this.updateClassTable(); // Update main table as well
        });

        // Color picker
        const color = this.dataManager.getClassColor(className);
        const colorDiv = document.createElement('div');
        colorDiv.className = 'color-cell';
        colorDiv.style.backgroundColor = this.rgbToHex(color.r, color.g, color.b);
        colorDiv.addEventListener('click', () => {
            this.showColorPicker(color, (newColor) => {
                this.dataManager.setClassColor(className, newColor);
                this.sceneManager.setClassColor(className, newColor);
                colorDiv.style.backgroundColor = this.rgbToHex(newColor.r, newColor.g, newColor.b);
                this.updateClassTable(); // Update main table as well
            });
        });

        row.appendChild(nameSpan);
        row.appendChild(visibleInput);
        row.appendChild(colorDiv);

        return row;
    }

    toggleGroupVisibility(groupName, classSet) {
        // Check if any classes in the group are currently visible
        let hasVisibleClasses = false;
        for (const className of classSet) {
            if (this.dataManager.isClassVisible(className)) {
                hasVisibleClasses = true;
                break;
            }
        }
        
        // Toggle: if any are visible, hide all; if all are hidden, show all
        const newVisibility = !hasVisibleClasses;
        
        let conflictCount = 0;
        for (const className of classSet) {
            // Check for conflicts with other groups
            const classGroupSet = this.classGroups.get(className);
            if (classGroupSet && classGroupSet.size > 1) {
                conflictCount++;
                console.log(`Visibility conflict resolved for "${className}" - applying group "${groupName}" settings (class belongs to ${classGroupSet.size} groups)`);
            }
            
            this.dataManager.setClassVisibility(className, newVisibility);
            this.sceneManager.setClassVisibility(className, newVisibility);
        }
        
        if (conflictCount > 0) {
            console.log(`Group "${groupName}" visibility operation completed with ${conflictCount} multi-group conflicts resolved via direct overwriting`);
        }
        
        // Update both main table and group display
        this.updateClassTable();
        this.updateGroupsDisplay();
    }

    setGroupColor(groupName, classSet) {
        // Get the color of the first class as the initial color
        const firstClassName = classSet.values().next().value;
        const initialColor = this.dataManager.getClassColor(firstClassName);
        
        this.showColorPicker(initialColor, (newColor) => {
            let conflictCount = 0;
            // Apply color to all classes in the group
            for (const className of classSet) {
                // Check for conflicts with other groups
                const classGroupSet = this.classGroups.get(className);
                if (classGroupSet && classGroupSet.size > 1) {
                    conflictCount++;
                    console.log(`Color conflict resolved for "${className}" - applying group "${groupName}" settings (class belongs to ${classGroupSet.size} groups)`);
                }
                
                this.dataManager.setClassColor(className, newColor);
                this.sceneManager.setClassColor(className, newColor);
            }
            
            if (conflictCount > 0) {
                console.log(`Group "${groupName}" color operation completed with ${conflictCount} multi-group conflicts resolved via direct overwriting`);
            }
            
            // Update both main table and group display
            this.updateClassTable();
            this.updateGroupsDisplay();
        });
    }

    toggleGroupBlur(groupName, classSet) {
        // Check if any classes in the group are currently blurred (have alpha < 1)
        let hasBlurredClasses = false;
        for (const className of classSet) {
            const color = this.dataManager.getClassColor(className);
            if (color.a !== undefined && color.a < 1) {
                hasBlurredClasses = true;
                break;
            }
        }
        
        // Toggle: if any are blurred, remove blur; if none are blurred, add blur
        const newAlpha = hasBlurredClasses ? 1.0 : 0.1;
        
        let conflictCount = 0;
        for (const className of classSet) {
            // Check for conflicts with other groups
            const classGroupSet = this.classGroups.get(className);
            if (classGroupSet && classGroupSet.size > 1) {
                conflictCount++;
                console.log(`Blur conflict resolved for "${className}" - applying group "${groupName}" settings (class belongs to ${classGroupSet.size} groups)`);
            }
            
            const color = this.dataManager.getClassColor(className);
            const newColor = { ...color, a: newAlpha };
            this.dataManager.setClassColor(className, newColor);
            this.sceneManager.setClassColor(className, newColor);
        }
        
        if (conflictCount > 0) {
            console.log(`Group "${groupName}" blur operation completed with ${conflictCount} multi-group conflicts resolved via direct overwriting`);
        }
        
        // Update both main table and group display
        this.updateClassTable();
        this.updateGroupsDisplay();
    }

    dissolveGroup(groupName) {
        // Confirm before dissolving
        if (!confirm(`Are you sure you want to dissolve the group "${groupName}"? All classes will remain visible but ungrouped.`)) {
            return;
        }
        
        // Get classes in the group
        const classSet = this.groups.get(groupName);
        if (!classSet) {
            console.warn(`Group "${groupName}" not found`);
            return;
        }
        
        // Remove group mappings for all classes in the group
        for (const className of classSet) {
            const classGroupSet = this.classGroups.get(className);
            if (classGroupSet) {
                classGroupSet.delete(groupName);
                // If this was the last group for this class, remove the entry entirely
                if (classGroupSet.size === 0) {
                    this.classGroups.delete(className);
                }
            }
        }
        
        // Remove the group itself
        this.groups.delete(groupName);
        
        // Update displays
        this.updateClassTable();
        this.updateGroupsDisplay();
        
        console.log(`Dissolved group "${groupName}" containing ${classSet.size} classes. Classes may still belong to other groups.`);
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    rgbToHex(r, g, b) {
        const toHex = (c) => {
            const hex = Math.round(c * 255).toString(16).padStart(2, '0');
            return hex;
        };
        return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    }
}
