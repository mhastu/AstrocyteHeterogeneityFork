function astro_geometry
% 2023-06-16
% Takes in the excel and matlab data and spits out a 3D
% image
%   The function loads the data set (if not found), find the radii
%   corresponding to the position indices, forms the branch information by
%   comparing the excel and matlab datas, and forms the 3D image. The
%   matlab_and_excel_data.mat includes
%       diameterData (combined from
%           H00_H2B_R_left_advanced_all_astro_Pt_Diameter_Type=Dendrite_Beginning.csv,
%           H00_H2B_R_left_advanced_all_astro_Pt_Diameter_Type=Dendrite_Branch.csv,
%           and
%           H00_H2B_R_left_advanced_all_astro_Pt_Diameter_Type=Dendrite_Terminal.csv
%           in the excel data)
%       positionData (renamed
%           H00_H2B_R_left_advanced_all_astro_Pt_Position.csv in the
%           excel data)
%       vFilamentsEdges (from the matlab data)
%       vFilamentsPoints (from the matlab data)
%   The function saves the branch data and 3D image in a .mat
%   branchData includes (each row in the cell array corresponds to a
%   section of branch between two intersections):
%       first column: the indices of the points in the matlab data (higher
%           resolution) the are part of the branch section (starting from
%           the one closest to the soma
%       second column: the indices of the "excel data" (lower res) that
%           define the starting and ending points of the branch section
%       third column: the depth of the branch section (distance from the
%           soma)
%       fourth column: the branch levels, i.e. the branch IDs (each whole
%           branch has it's own level from start to tip. New branch with
%           its own level begins at each branching point, where the
%           daughter branch with larger mean diameter retains the level of
%           the mother branch, or if the diameters are equal, then the
%           smaller branching angle)
%       fifth column: the "ancestor" branches, i.e. the levels of the
%           branches where the current branch level originates from all the
%           way until the soma

% folder i2M and function stlwrite2 have to be in the same location


addpath(genpath('i2M'))

%% settings

% select the cell to plot (1-302) (all cell won't work, including 44, 83,
% 87, 89, 90, 91, 102, 120, 121, 124, 125, 130, 150, 179, 188, 203, 227,
% 263, 277, 280, 286, 287, 291, 292, 293, 294, 295, 296, 297, 298, 299,
% 300, 301, 302)

% 1 to create the 3D figure, 0 to no
makeFig = 0;

% resulotion in um
resolution = 0.15;

% step size used to draw the branches in um
drawingStep = 0.1;

% extracellular space thickenss in um
ecR = 2;

% restrictions on the "drawing"
levelLimit = 5;
depthLimit = 1000; % maximum depth for the branches
radiusLimit = 0; % minimum radius for the branches
% use the draw the first, second, etc. child branches for a branch, each
% line contains the "distance" to the ancestor and the ancestor level. E.g.
% line 1 1 is the first children of branch level 1. REPLACE WITH AN EMPTY
% MATRIX IF NOT USED
childBranches = [];

[file,path] = uigetfile('*.mat',"Select the mat file for the data set");
if file == 0
    return
end

load(strcat(path,file))

datasetName = input("Give the dataset name: ","s");

cellID = input("Give the cell ID (maximum ID is " + num2str(length(vFilamentsPoints)) + "): ");

if ~(cellID <= length(vFilamentsPoints))
   disp("Too large cell ID!")
   return
end

userNameInput = input('Give extra information for the export name (name will be "cell_<dataset>_<number>_<your input here>": ', "s");
if strcmp(userNameInput,'')
    name = ['cell_' datasetName '_' num2str(cellID)];
else
    name = ['cell_' datasetName '_' num2str(cellID) '_' userNameInput];
end

mkdir(name);

%% get the radii corresponding to each point

% message
fprintf('Connecting positions with the radii...\n');

% get the points that are part of the same cell
sameCell = positionData.FilamentID == positionData.FilamentID(cellID);

% get the coordinates for these points
excelCoord = [positionData.PtPositionX(sameCell) positionData.PtPositionY(sameCell) positionData.PtPositionZ(sameCell)];

% get the IDs for these points
pointIDs = positionData.ID(sameCell);

% initialize a vector for the radii corresponding to the points
correspondingDiameters = zeros(size(pointIDs));

% go through the points
for i = 1:length(pointIDs)
    
    % find the diameter in the diameter data that has the same ID as the
    % position coordinate of the same vector index
    correspondingDiameters(i) = diameterData.PtDiameter(diameterData.ID == pointIDs(i));
    
end

%% gather the information on the branches

% message
fprintf('Forming the branch data...\n');

connections = vFilamentsEdges{cellID};
finePoints = vFilamentsPoints{cellID};

% set the index to start from 1
if connections(1,1) == 0
    connections = connections + 1;
end

correspondingIdx = zeros(size(excelCoord,1),1);

% find the indices of the excelCoords in the matCoords
for i = 1:size(excelCoord,1)
    [~,correspondingIdx(i)] = min((excelCoord(i,1) - finePoints(:,1)).^2 + (excelCoord(i,2) - finePoints(:,2)).^2 + (excelCoord(i,3) - finePoints(:,3)).^2);
end

% find connections where the soma (the finePoint with index 1) has a part
% and save them to branches to check
branches2Check = connections(connections(:,1) == 1,:);

% add branch depths as the third column
branches2Check = [branches2Check ones(size(branches2Check,1),1)];

% cell structure to store the branch data
branchData = cell(0,0);

i = 1;

% go through the branches to check
while 1
    
    currentBranch = branches2Check(i,1:2);
    
    % continue until a point in the excelPoints is hit
    while 1
        
        % get the connections where the current branch end is the starting
        % point
        ii = find(connections(:,1) == currentBranch(end));
        
        % if the end point of the current branch belongs to the
        % excelPoints (is a branching point)
        if any(currentBranch(end) == correspondingIdx)
            
            % get the starting and ending points of the branch in the excel
            % data
            startIdx = find(correspondingIdx == currentBranch(1));
            endIdx = find(correspondingIdx == currentBranch(end));
            
            % add the branch to the saved data (the indices in the finer
            % data, the indices in the excel data, and the branch depth)
            branchData(end+1,:) = {currentBranch , [startIdx,endIdx] , branches2Check(i,3)}; %#ok<SAGROW>
            
            % add the connections that have the ending point of this branch
            % as their starting point into the branches2Check (and their
            % depth)
            branches2Check = [branches2Check ; [connections(ii,:) (branches2Check(i,3)+1).*ones(length(ii),1)]]; %#ok<AGROW>
            
            break
        else
            
            % if not, add the end point of the connection to the current
            % branch
            currentBranch(end+1) = connections(ii,2); %#ok<SAGROW>
            
        end
    end
    
    % next branch
    i = i + 1;
    
    % if no more branches
    if i > size(branches2Check,1)
        break
    end
end

%% find the branch levels (ID for the branch)

% message
fprintf('Getting branch levels...\n');

branchLevelChecked = zeros(size(branchData,1),1);

branchLevels = zeros(size(branchData,1),1);

% get the vectors for the first and last fine segment for each branch
% section

firstSegment = zeros(size(branchData,1),3);
lastSegment = zeros(size(branchData,1),3);

for i = 1:size(branchData,1)
    firstSegment(i,:) = finePoints(branchData{i,1}(2),:) - finePoints(branchData{i,1}(1),:);
    lastSegment(i,:) = finePoints(branchData{i,1}(end),:) - finePoints(branchData{i,1}(end-1),:);
end

% get the branch starting and ending indices for all branches
branchDataTemp = cell2mat(branchData(:,2));

% the branches starting from the soma point
connected2Soma = branchDataTemp(branchDataTemp(:,1) == 1,:);

% get the average diameter for these branches
diametersTemp = mean(reshape(correspondingDiameters(connected2Soma(:)),[],2),2);

% sort the diameters and get new indices
[~,idx] = sort(diametersTemp,1,'descend');

% set the branchlevels of the sections that start from the soma according
% to their thickness
branchLevels(branchDataTemp(:,1) == 1) = idx;

% create the branchAncestors cell
branchAncestors = mat2cell(branchLevels,ones(size(branchLevels)));

% get current maximum level
currentMax = max(branchLevels);

% loop until all branch sections have been checked
while sum(branchLevelChecked) ~= length(branchLevelChecked)
    
    % get the current maximum as a temp
    currentMaxTemp = currentMax;
    
    % go through the branch sections that already have a level designated
    % to them
    for i = 1:currentMaxTemp
        
        % check the branch sections that have the branch level i and whose
        % daughters have not been checked yet
        motherBranch = find(branchLevels == i & branchLevelChecked == 0);
        
        % check if they exist (e.g. if end has been reached already)
        if numel(motherBranch) > 0
            
            % get the endpoint of the mother branch section
            motherBranchEndPoint = branchDataTemp(motherBranch,2);
            
            % set the branch level checked to true
            branchLevelChecked(motherBranch) = 1;
            
            % find the branch sections whose starting point is the same as
            % the endpoint of the mother
            daughterBranches = find(branchDataTemp(:,1) == motherBranchEndPoint);
            
            % check if they exist
            if numel(daughterBranches) > 0
                
                % get the number of daughters
                nDaughters = length(daughterBranches);
                
                % get the branchData for the daughters
                branchDataTempTemp = branchDataTemp(daughterBranches,:);
                
                % get the mean diameters for the daughter sections
                diametersTemp = mean(reshape(correspondingDiameters(branchDataTempTemp(:)),[],2),2);
                
                % sort the diameters
                [diametersTemp,idx] = sort(diametersTemp,1,'descend');
                
                % reorder the daughters
                daughterBranches = daughterBranches(idx);
                
                % get the unique diameters
                uniques = unique(diametersTemp,'stable');
                
                % the levels available for the daughters, including the
                % mother level
                availableLevels = [i currentMax+1:currentMax+nDaughters-1];
                
                % temp index to keep track of levels
                levelIdx = 1;
                
                % go through the unique diameters
                for j = 1:length(uniques)
                    
                    % find the daughters that have this diameter
                    temp = diametersTemp == uniques(j);
                    
                    % if there is only one
                    if sum(temp) == 1
                        
                        % add the level for the daughter
                        branchLevels(daughterBranches(temp)) = availableLevels(levelIdx);
                        
                        % add the branch ancestor data
                        if ~(levelIdx == 1)
                            branchAncestors{daughterBranches(temp)} = [branchAncestors{motherBranch} availableLevels(levelIdx)];
                        else
                            branchAncestors{daughterBranches(temp)} = branchAncestors{motherBranch};
                        end
                        
                        levelIdx = levelIdx + 1;
                        
                    else
                        
                        % find the daughters with the same diameter
                        sameDiameter = find(temp);
                        
                        % vector for the angles
                        angles = zeros(sum(temp),1);
                        
                        % get the last segment of the mother branch as a
                        % vector
                        v1 = lastSegment(motherBranch,:);
                        
                        % go through the daughters, calculate the first
                        % segment vectors and calculate the angles
                        for k = 1:length(angles)
                            v2 = firstSegment(daughterBranches(sameDiameter(k)),:);
                            angles(k) = acosd(dot(v1 / norm(v1), v2 / norm(v2)));
                        end
                        
                        % sort the angles
                        [~,idx] = sort(angles);
                        
                        % go through the daughters with the same diameter
                        for k = 1:length(angles)
                            
                            % add the level for the daughter
                            branchLevels(daughterBranches(sameDiameter(idx(k)))) = availableLevels(levelIdx);
                            
                            % add the branch ancestor data
                            if ~(levelIdx == 1)
                                branchAncestors{daughterBranches(sameDiameter(idx(k)))} = [branchAncestors{motherBranch} availableLevels(levelIdx)];
                            else
                                branchAncestors{daughterBranches(sameDiameter(idx(k)))} = branchAncestors{motherBranch};
                            end
                            levelIdx = levelIdx + 1;
                            
                        end
                        
                    end
                end
            end
        end
        
        % get the current max level
        currentMax = max(branchLevels);
        
    end
    
end

branchLevels = mat2cell(branchLevels,ones(size(branchLevels)));

branchData = [branchData branchLevels branchAncestors];

%save(['./' name '/' name '_branch_data.mat'], 'branchData');
save(['./' name '/' 'branchData.mat'], 'branchData'); %modification for better execution of the python code

%% create the 3D geometry

if makeFig
    
    % message
    fprintf('Creating the 3D image...\n');
    
    % padding around the cell
    padding = 3;
    
    % maximum and minimum coordinates in each direction with the padding
    xMin = round(min(finePoints(:,1)) - padding);
    xMax = round(max(finePoints(:,1)) + padding);
    yMin = round(min(finePoints(:,2)) - padding);
    yMax = round(max(finePoints(:,2)) + padding);
    zMin = round(min(finePoints(:,3)) - padding);
    zMax = round(max(finePoints(:,3)) + padding);
    
    % vectors for the box positions in each direction
    boxX = xMin:resolution:xMax;
    boxY = yMin:resolution:yMax;
    boxZ = zMin:resolution:zMax;
    
    % meshgrid from the vectors in 3D (x and y flipped since the indexing goes
    % line-wise first to retain x as the horizontal coordinate)
    [boxY,boxX,boxZ] = meshgrid(boxY,boxX,boxZ);
    
    % get the image size
    imageSize = size(boxX);
    
    % transform into linear format
    xLin = boxX(:);
    yLin = boxY(:);
    zLin = boxZ(:);
    
    % make the image also as a linear vector
    image = zeros(size(xLin));
    
    % string for the loop progress print
    reverseStr = '';
    
    
%     for i = size(branchData,1):-1:1
%         if branchData{i,4} > levelLimit
%             if numel(childBranches) > 0
%                 for ii = 1:size(childBranches,1)
%                     if length(branchData{i,5} > childBranches(i,1) && childBranches(i,1)
%                         
%                     end
%                 end
%             else
%                 branchData(i,:) = [];
%             end
%         end
%     end
    
    numberOfBranches2Draw = size(branchData,1);
    currentBranch = 1;
    % go through the branches
    for i = 1:size(branchData,1)
        
        % print the loop progress
        msg = sprintf('Drawn branches: %d/%d', currentBranch, numberOfBranches2Draw);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        
        childDrawn = false;
        if numel(childBranches) > 0
            for ii = 1:size(childBranches,1)
                if length(branchData{i,5}) > childBranches(ii,1) && branchData{i,5}(end-childBranches(ii,1)) == childBranches(ii,2)
                    childDrawn = true;
                    break
                end
            end
        end
        
        % if below limit
        if branchData{i,3} <= depthLimit && (branchData{i,4} <= levelLimit || childDrawn)
            
            % get the indices of the points in the finer matlab data for this
            % branch (start and end points of all small sections)
            P1 = branchData{i,1}(1:end-1);
            P2 = branchData{i,1}(2:end);
            
            % get the coordinates for the small section start and end points
            P1X = finePoints(P1,1);
            P2X = finePoints(P2,1);
            P1Y = finePoints(P1,2);
            P2Y = finePoints(P2,2);
            P1Z = finePoints(P1,3);
            P2Z = finePoints(P2,3);
            
            % get the branch section lengths
            branchSectionLengths = sqrt((P1X - P2X).^2 + (P1Y - P2Y).^2 + (P1Z - P2Z).^2);
            
            % calculate the tolal branch length
            totalBranchLength = sum(branchSectionLengths);
            
            % get the radii at the start and at the end of the branch
            startR = correspondingDiameters(branchData{i,2}(1))/2;
            endR = correspondingDiameters(branchData{i,2}(2))/2;
            
            % if the mean radius is above the radius limit
            if mean([startR endR]) >= radiusLimit
                
                % calculate the cumulative length of the branch sections from
                % the start
                cumulativeLengths = cumsum(branchSectionLengths);
                
                % interpolate the calculate the radii at each intermediate
                % branch point
                intermediateRadii = (endR - startR)/(totalBranchLength).*(cumulativeLengths(1:end-1)) + startR;
               
                % combine all the radii
                allRadii = [startR intermediateRadii' endR];
                
                maxRadii = max(allRadii);
                
                xP = [P1X; P2X];
                yP = [P1Y; P2Y];
                zP = [P1Z; P2Z];
                
                xMin = min(xP)-maxRadii; xMax = max(xP)+maxRadii;
                yMin = min(yP)-maxRadii; yMax = max(yP)+maxRadii;
                zMin = min(zP)-maxRadii; zMax = max(zP)+maxRadii;
                
                close = find((xLin > xMin).*(xLin < xMax).*(yLin > yMin).*(yLin < yMax).*(zLin > zMin).*(zLin < zMax));
                
                % go through the points in the branch
                for j = 1:length(P1)
                    
                    % find the pixels in the image that are closer than the
                    % radius of the starting point of the branch section and
                    % assign 1 to them
                    
                    closePixels = find_close_pixels(P1X(j),P1Y(j),P1Z(j),allRadii(j),xLin(close),yLin(close),zLin(close));
                    
                    image(close(closePixels)) = 1;
                     
                    % initialize the increment distance
                    incrementDistance = drawingStep;
                    
                    % step along the branch section until the section length
                    % has been passed
                    while incrementDistance < branchSectionLengths(j)
                        
                        % find the intermediate coordinates and radius along
                        % the branch section by interpolation
                        interX = (P2X(j) - P1X(j))/(branchSectionLengths(j)).*(incrementDistance) + P1X(j);
                        interY = (P2Y(j) - P1Y(j))/(branchSectionLengths(j)).*(incrementDistance) + P1Y(j);
                        interZ = (P2Z(j) - P1Z(j))/(branchSectionLengths(j)).*(incrementDistance) + P1Z(j);
                        interR = (allRadii(j+1) - allRadii(j))/(branchSectionLengths(j)).*(incrementDistance) + allRadii(j);
                        %                     interR = endR;
                        
                        % find the pixels in the image that are closer than the
                        % radius of the intermediate point of the branch
                        % section assign 1 to them
                        closePixels = find_close_pixels(interX,interY,interZ,interR,xLin(close),yLin(close),zLin(close));
                        
                        image(close(closePixels)) = 1;
                        
                        % increase the intermediate distance by the step
                        incrementDistance = incrementDistance + drawingStep;
                        
                    end
                    
                    % find the pixels in the image that are closer than the
                    % radius of the end point of the branch section and assign
                    % 1 to them
                    image = image + find_close_pixels(P2X(j),P2Y(j),P2Z(j),allRadii(j+1),xLin,yLin,zLin);
                    
                end
                currentBranch = currentBranch + 1;
            else
                numberOfBranches2Draw = numberOfBranches2Draw - 1; 
            end
        else
            numberOfBranches2Draw = numberOfBranches2Draw - 1;
        end
    end
    
    % to move to new line after progress printing
    fprintf('\n');
    
    % reshape the image back to 3D
    image3D = reshape(image,imageSize(1),imageSize(2),imageSize(3));
    
    % cut off extra padding
    [r,c,v] = ind2sub(size(image3D),find(image3D));
    maxR = max(r); minR = min(r);
    maxC = max(c); minC = min(c);
    maxV = max(v); minV = min(v);
    
    image3D(:,:,maxV+1:end) = [];
    image3D(:,maxC+1:end,:) = [];
    image3D(maxR+1:end,:,:) = [];
    image3D(:,:,1:minV-1) = [];
    image3D(:,1:minC-1,:) = [];
    image3D(1:minR-1,:,:) = [];
    
    % save the image into a mat file
    save(['./' name '/' name '_image.mat'], 'image3D');
    
    % save the image as a multilayer tiff
    imwrite(image3D(:,:,1), ['./' name '/' name '.tiff']);
    for i = 2:size(image3D,3)
        imwrite(image3D(:,:,i), ['./' name '/' name '.tiff'],'writemode', 'append');
    end
    
    [node,elem] = vol2surf(image3D>0.5,1:size(image3D,1),1:size(image3D,2),1:size(image3D,3),1,'cgalsurf');
    stlwrite2(['./' name '/' name '_cell.stl'],elem(:,1:3),node,'MODE','ascii');
    
    imageEC = padarray(image3D,ceil(ecR/resolution).*[1 1 1],0,'both');
    
    se = strel('sphere',ceil(ecR/resolution));
    imageEC = imdilate(logical(imageEC),se);
    
    [node,elem] = vol2surf(imageEC>0.5,1:size(imageEC,1),1:size(imageEC,2),1:size(imageEC,3),4,'cgalsurf');
    stlwrite2(['./' name '/' name '_extracellular.stl'],elem(:,1:3),node,'MODE','ascii');
    
end

end

% function to find the pixels that are closer than the radius
function closeImage = find_close_pixels(pointX,pointY,pointZ,radius,gridX,gridY,gridZ)

% check which distances between pixel center and the point on the
% branch are closer than the radius
closeImage = (pointX - gridX).^2 + (pointY - gridY).^2 + (pointZ - gridZ).^2 <= radius^2;

end
