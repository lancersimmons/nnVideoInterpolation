-- Video Interpolation CNN
-- Fall, 2016


-- REQUIRES:
require('image')
require('nn')
require('optim')
require('math')
load_images = require('load_images')
require 'torch'
require 'cutorch'
require 'cunn'

math.randomseed( os.time() )

-- PARAMETERS:

-- Count frames in frames dir
local i, t, popen = 0, {}, io.popen
local pfile = popen('ls -a frames')
for filename in pfile:lines() do
    i = i + 1
    t[i] = filename
end
pfile:close()

-- t should be reduced by 2 to get rid of . and ..
t = t - 2
numberOfInputFramesToUse = t

modelToLoadFromFile = nil
if arg[1] ~= nil then
	loadModelFlag = true
	modelToLoadFromFile = arg[1]
else
	loadModelFlag = false
end

--3 for triplets, 5 for quintuplets, etc.
frameGroupSize = 3

-- ARCHITECTURAL PARAMETERS
trainingCycles = 10
batch_size = 1
frame_width = 160
frame_height = 96
n_channels = 3
torch.setnumthreads(1)


-- CODE START:
print("Start script...")

-- set default tensor type to 32-bit float
torch.setdefaulttensortype('torch.FloatTensor')

-- set up the rng for math lib
--math.randomseed(os.time())

-- open frames directory, count frames
fdir = io.popen('ls frames')
frameCount = 0
for name in fdir:lines() do 
	-- print(name) 
	frameCount = frameCount + 1
end

print(frameCount, " frames in directory.")
print("Loading frames into tensor...")

-- tensor of all frames in frames directory
allFrames = load_images.load('frames', frameCount)

-- confirm that parameter is correct size
if numberOfInputFramesToUse > allFrames:size(1) then
	print "Bad parameter: numberOfInputFramesToUse > number of frames"
	os.exit()
end

-- if ((numberOfInputFramesToUse % frameGroupSize) ~= 0) then
-- 	print "Bad parameter: numberOfInputFramesToUse % frameGroupSize ~= 0"
-- 	os.exit()
-- end

-- if ((numberOfInputFramesToUse % 10) ~= 0) then
-- 	print "Bad parameter: numberOfInputFramesToUse % 10 ~= 0"
-- 	os.exit()
-- end

-- Set size of training set, test set
sizeOfTrainingSet = (math.floor((9/10) * numberOfInputFramesToUse))
sizeOfTestSet = numberOfInputFramesToUse - sizeOfTrainingSet

print("Training Set Size: ", sizeOfTrainingSet)
print("Test Set Size:     ", sizeOfTestSet)

--  build training set
trainingFrames = 
	torch.Tensor(
	sizeOfTrainingSet,
	allFrames:size(2),	-- channels (3)
	allFrames:size(3),	-- height (96)
	allFrames:size(4))	-- width (160)

for frameNumber=1, sizeOfTrainingSet, 1
do
	trainingFrames[frameNumber] = allFrames[frameNumber]
end


-- table filled with all the triplet leading frame indices
-- used for random sampling of triplets during training

train_triplets = torch.Tensor(sizeOfTrainingSet-2, 3, allFrames:size(2), allFrames:size(3), allFrames:size(4))
aug_train_triplets = torch.Tensor((sizeOfTrainingSet-2)*4, 3, allFrames:size(2), allFrames:size(3), allFrames:size(4))

for i=1, sizeOfTrainingSet-2 do
	train_triplets[i][1] = trainingFrames[i]
	train_triplets[i][2] = trainingFrames[i+1]
	train_triplets[i][3] = trainingFrames[i+2]
end

shuffle_data = torch.randperm(train_triplets:size(1))

-- Data as is into augmentation set

for i=1, sizeOfTrainingSet-2 do
	temp = shuffle_data[i]
	aug_train_triplets[temp][1] = train_triplets[temp][1] 
	aug_train_triplets[temp][2] = train_triplets[temp][2] 
	aug_train_triplets[temp][3] = train_triplets[temp][3] 
end

-- Reverse order triplets

for i=1, sizeOfTrainingSet-2 do 
	temp = shuffle_data[i]
	aug_train_triplets[temp + sizeOfTrainingSet-2][1] = train_triplets[temp][3] 
	aug_train_triplets[temp + sizeOfTrainingSet-2][2] = train_triplets[temp][2] 
	aug_train_triplets[temp + sizeOfTrainingSet-2][3] = train_triplets[temp][1] 
end

-- Vfliped triplets

for i=1, sizeOfTrainingSet-2 do 
	temp = shuffle_data[i]
	image.vflip(aug_train_triplets[temp + 2 * (sizeOfTrainingSet-2)][1], train_triplets[temp][1])
	image.vflip(aug_train_triplets[temp + 2 * (sizeOfTrainingSet-2)][2], train_triplets[temp][2])
	image.vflip(aug_train_triplets[temp + 2 * (sizeOfTrainingSet-2)][3], train_triplets[temp][3])
end

-- Hfliped triplets

for i=1, sizeOfTrainingSet-2 do 
	temp = shuffle_data[i]
	image.hflip(aug_train_triplets[temp + 3 * (sizeOfTrainingSet-2)][1], train_triplets[temp][1])
	image.hflip(aug_train_triplets[temp + 3 * (sizeOfTrainingSet-2)][2], train_triplets[temp][2])
	image.hflip(aug_train_triplets[temp + 3 * (sizeOfTrainingSet-2)][3], train_triplets[temp][3])
end


--  build testing set
testingFrames = 
	torch.Tensor(
	sizeOfTestSet,
	allFrames:size(2),
	allFrames:size(3),
	allFrames:size(4))

for frameNumber=1, sizeOfTestSet, 1
do
	-- testingFrames[frameNumber] = allFrames[frameNumber + sizeOfTrainingSet]:clone()
	testingFrames[frameNumber] = allFrames[frameNumber + sizeOfTrainingSet]
end


-- -- Scale RGB from [0..1] to [-1..1] to satisfy Tanh
aug_train_triplets = aug_train_triplets * 2
aug_train_triplets = aug_train_triplets - 1

testingFrames = testingFrames * 2
testingFrames = testingFrames - 1

if loadModelFlag then
	print("Loading model from file...")
	model = torch.load(modelToLoadFromFile)
else
	------------ Model ------------------------------------------------

	criterion = nn.AbsCriterion()
	criterion = criterion:cuda()
	s1 = nn.Sequential()
	s00 = nn.Sequential()
	s01 = nn.Sequential()
	s2 = nn.Sequential()
	s3 = nn.Sequential()
	s4 = nn.Sequential()
	s5 = nn.Sequential()
	s6 = nn.Sequential()
	s7 = nn.Sequential()
	s8 = nn.Sequential()
	s9 = nn.Sequential()
	s10 = nn.Sequential()
	model = nn.Sequential()

	c1 = nn.Parallel(1, 1)
	c2 = nn.DepthConcat(1)
	c3 = nn.DepthConcat(1)
	c4 = nn.DepthConcat(1)
	c5 = nn.DepthConcat(1)
	c6 = nn.DepthConcat(1)

	s1:add(nn.SpatialConvolution(3, 96, 3, 3, 1, 1, 1, 1):cuda()) ----- frame 1 conv block 1
	s1:add(nn.Tanh():cuda())
	s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda())
	s1:add(nn.Tanh():cuda())
	--s1:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 96 x fr_w x fr_h

	s2:add(nn.SpatialConvolution(3, 96, 3, 3, 1, 1, 1, 1):cuda()) ----- frame 3 conv block 1
	s2:add(nn.Tanh():cuda())
	s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda())
	s2:add(nn.Tanh():cuda())
	--s2:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 96 x fr_w x fr_h


	c1:add(s1)
	c1:add(s2)											--- ouput_size: (96+96) x fr_w x fr_h

	s00:add(c1)
	s00:add(nn.SpatialConvolution(192, 96, 3, 3, 1, 1, 1, 1):cuda()) ------ frames conv block 1
	s00:add(nn.Tanh():cuda())
	s00:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda())
	s00:add(nn.Tanh():cuda())	
	s00:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 96 x fr_w/2 x fr_h/2

	s01:add(s00)
	s01:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda()) ------ frames conv block 2
	s01:add(nn.Tanh():cuda())
	s01:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda())
	s01:add(nn.Tanh():cuda())	
	s01:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 96 x fr_w/4 x fr_h/4


	s3:add(s01)								--------------------- Conv block 3
	s3:add(nn.SpatialConvolution(96, 128, 3, 3, 1, 1, 1, 1):cuda()) 
	s3:add(nn.Tanh():cuda())
	s3:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda())
	s3:add(nn.Tanh():cuda())
	s3:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 128 x fr_w/8 x fr_h/8


	s4:add(s3)								--------------------- Conv block 4
	s4:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda()) 
	s4:add(nn.Tanh():cuda())
	s4:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda())
	s4:add(nn.Tanh():cuda())
	s4:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 128 x fr_w/16 x fr_h/16


	s5:add(s4)								--------------------- Conv block 5
	s5:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda()) 
	s5:add(nn.Tanh():cuda())
	s5:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda())
	s5:add(nn.Tanh():cuda())
	s5:add(nn.SpatialMaxPooling(2, 2, 2, 2):cuda())		--- ouput_size: 128 x fr_w/32 x fr_h/32


	s6:add(s5)								--------------------- DeConv block 1
	s6:add(nn.SpatialFullConvolution(128, 128, 4, 4, 2, 2, 1, 1):cuda())
	s6:add(nn.Tanh():cuda())
	s6:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda()) 
	s6:add(nn.Tanh():cuda())									--- ouput_size: 128 x fr_w/16 x fr_h/16

	c2:add(s4)
	c2:add(s6)											--- ouput_size: (128+128) x fr_w/16 x fr_h/16

	s7:add(c2)								--------------------- DeConv block 2
	s7:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1):cuda())
	s7:add(nn.Tanh():cuda())
	s7:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda()) 
	s7:add(nn.Tanh():cuda())									--- ouput_size: 128 x fr_w/8 x fr_h/8

	c3:add(s3)
	c3:add(s7)											--- ouput_size: (128+128) x fr_w/16 x fr_h/16

	s8:add(c3)								--------------------- DeConv block 3
	s8:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1):cuda())
	s8:add(nn.Tanh():cuda())
	s8:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1):cuda()) 
	s8:add(nn.Tanh():cuda())									--- ouput_size: 128 x fr_w/4 x fr_h/4

	c4:add(s01)
	c4:add(s8)											--- ouput_size: (192+128) x fr_w/16 x fr_h/16

	s9:add(c4)								--------------------- DeConv block 4
	s9:add(nn.SpatialFullConvolution(224, 96, 4, 4, 2, 2, 1, 1):cuda())
	s9:add(nn.Tanh():cuda())
	s9:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda()) 
	s9:add(nn.Tanh():cuda())									--- ouput_size: 96 x fr_w/2 x fr_h/2

	c5:add(s00)
	c5:add(s9)

	s10:add(c5)								--------------------- DeConv block 5
	s10:add(nn.SpatialFullConvolution(192, 96, 4, 4, 2, 2, 1, 1):cuda())
	s10:add(nn.Tanh():cuda())
	s10:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1):cuda()) 
	s10:add(nn.Tanh():cuda())									--- ouput_size: 96 x fr_w/1 x fr_h/1

	c6:add(c1)
	c6:add(s10)

	model:add(c6)							--------------------- Output Layer
	model:add(nn.SpatialConvolution(288, 3, 3, 3, 1, 1, 1, 1):cuda())
	model:add(nn.Tanh():cuda())									--- ouput_size: 3 x fr_w/1 x fr_h/1
	model = model:cuda()
	-------------------------------------------------------------------
end


-- these tensors will hold the outer and inner frames of the triplets, respectively
input_frames = torch.Tensor(2, n_channels, frame_height, frame_width)
input_frames = input_frames:cuda()
output_frames_middle = torch.Tensor(n_channels, frame_height, frame_width)
output_frames_middle = output_frames_middle:cuda()


if loadModelFlag then
	print("Skipping training...")
else
-- TRAINING
print("Forwarding...")

-- for training iterations
local theta, gradTheta = model:getParameters()
local optimState = {learningRate = 0.01, momentum = 0, learningRateDecay = 5e-7}model:training()

for k = 1, 15 do
	epochloss = 0
	for loopIterator=1, (sizeOfTrainingSet-2)*4, 1
	do
		--print("Training loop:", loopIterator)
		function feval(theta)
			-- get input, output frames (first, last, middle)
			
			input_frames[1] = aug_train_triplets[loopIterator][1] -- first
			input_frames[2] = aug_train_triplets[loopIterator][3] -- last			
		
			output_frames_middle = aug_train_triplets[loopIterator][2] -- middle
			output_frames_middle=output_frames_middle:cuda()

			-- forward data and print loss
			local h_x=model:forward(input_frames)
			--print(output_frames_middle:type())
			local J=criterion:forward(h_x, output_frames_middle)
			io.write(loopIterator, " ", J, "\r")
			io.flush()
			epochloss = epochloss + J
			gradTheta:zero()
			local dJ_dh_x = criterion:backward(h_x, output_frames_middle)
			dJ_dh_x:cuda()
			model:backward(input_frames, dJ_dh_x)
			return J, gradTheta
		end
		optim.adagrad(feval, theta, optimState)
	end
	print("epoch loss:", epochloss/((sizeOfTrainingSet-2)*4), "epoch done:", k)
end

torch.save("myModel", model)

end


-- TESTING
print("Testing...")

odd = 1
odd_original = 1
even = 2
for loopIterator=1, sizeOfTestSet-2, 1
do
	-- get input frames, output frame (first, last, middle)
	input_frames[1] = testingFrames[loopIterator] -- first
	input_frames[2] = testingFrames[loopIterator + 2] -- last
	
	groundTruthFrame = torch.Tensor(n_channels, frame_height, frame_width)
	groundTruthFrame = testingFrames[loopIterator+1]
	
	predictedFrame = model:forward(input_frames)

	tempPredictedImage = predictedFrame
	tempGroundTruth = groundTruthFrame

	-- -- Scale RGB from [-1..1] to [0..1] to output correctly
	tempPredictedImage = tempPredictedImage + 1
	tempPredictedImage = tempPredictedImage / 2

	tempGroundTruth = tempGroundTruth + 1
	tempGroundTruth = tempGroundTruth / 2


	filename = string.format('pics/original/OriginalSeq%05d.jpg', odd_original)
	-- filename = "pics/original/OriginalSeq" .. tostring(odd_original).. ".jpg"
	image.save(filename, tempGroundTruth)
	odd_original = odd_original + 1

	filename = string.format('pics/prediction/PredictedSeq%05d.jpg', odd)
	-- filename = "pics/prediction/PredictedSeq" .. tostring(odd).. ".jpg"
	image.save(filename, tempGroundTruth)
	odd = odd + 2

	filename = string.format('pics/prediction/PredictedSeq%05d.jpg', even)
	-- filename = "pics/prediction/PredictedSeq" .. tostring(even).. ".jpg"
	image.save(filename, tempPredictedImage)
	even = even + 2
end

-----------------------------------------------------------------------------------------------

--[[tripletLeadingIndices = {}
for i=1, sizeOfTrainingSet-2 do
  tripletLeadingIndices[i] = i
end

-- shuffle the index leading table entries
for i = sizeOfTrainingSet-2, 1, -1 do -- backwards
    local r = math.random(i) -- select a random number between 1 and i
    -- swap the randomly selected item to position i
    tripletLeadingIndices[i], tripletLeadingIndices[r] = tripletLeadingIndices[r], tripletLeadingIndices[i] 
end]]--

-- -- Sanity check
-- for i=1, #tripletLeadingIndices, 1 do
-- 	print(i, tripletLeadingIndices[i])
-- end
-- print("len tripletLeadingIndices:", #tripletLeadingIndices)


--[[
local shuffle_data = torch.randperm(trainingFrame:size(1))

trainingFrames = torch.Tensor(
	sizeOfTrainingSet*3,
	allFrames:size(2),	-- channels (3)
	allFrames:size(3),	-- height (96)
	allFrames:size(4))

for i=1, sizeOfTrainingSet do
	temp=shuffle_data[i]
	trainingFrames[i]=trainingFrame[temp]
end

for i=1, sizeOfTrainingSet, 1 do
	temp=shuffle_data[i]
	--print(temp)
	image.vflip(trainingFrames[i+sizeOfTrainingSet], trainingFrame[temp])
end

for i=1, sizeOfTrainingSet, 1 do
	temp=shuffle_data[i]
	--print(temp)
	image.hflip(trainingFrames[i+sizeOfTrainingSet*2], trainingFrame[temp])
end
]]--

