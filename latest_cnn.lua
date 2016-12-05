-- Video Interpolation CNN
-- Fall, 2016


-- REQUIRES:
require('image')
require('nn')
require('optim')
require('math')
load_images = require('load_images')

-- PARAMETERS:
numberOfInputFramesToUse = 300

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
sizeOfTrainingSet = math.floor((9/10) * numberOfInputFramesToUse)
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
	-- trainingFrames[frameNumber] = allFrames[frameNumber]:clone()
	trainingFrames[frameNumber] = allFrames[frameNumber]
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



-- -- FIXME: Why is this causing the loss to be worse?
-- -- Scale RGB from [0..1] to [-1..1] to satisfy Tanh
trainingFrames = trainingFrames * 2
trainingFrames = trainingFrames - 1

testingFrames = testingFrames * 2
testingFrames = testingFrames - 1




-- FIXME: Fix the Tanh/relu scaling issue
------------ Model ------------------------------------------------

criterion = nn.AbsCriterion()

s1 = nn.Sequential()
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

s1:add(nn.SpatialConvolution(3, 96, 3, 3, 1, 1, 1, 1)) ----- frame 1 conv block 1
s1:add(nn.Tanh())
s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s1:add(nn.Tanh())
s1:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/2 x fr_h/2

s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) ------ frame 1 conv block 2
s1:add(nn.Tanh())
s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s1:add(nn.Tanh())	
s1:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/4 x fr_h/4

s2:add(nn.SpatialConvolution(3, 96, 3, 3, 1, 1, 1, 1)) ----- frame 3 conv block 1
s2:add(nn.Tanh())
s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s2:add(nn.Tanh())
s2:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/2 x fr_h/2

s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) ----- frame 3 conv block 2
s2:add(nn.Tanh())
s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s2:add(nn.Tanh())
s2:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/4 x fr_h/4

c1:add(s1)
c1:add(s2)											--- ouput_size: (96+96) x fr_w/4 x fr_h/4


s3:add(c1)								--------------------- Conv block 3
s3:add(nn.SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1)) 
s3:add(nn.Tanh())
s3:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
s3:add(nn.Tanh())
s3:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 128 x fr_w/8 x fr_h/8


s4:add(s3)								--------------------- Conv block 4
s4:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s4:add(nn.Tanh())
s4:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
s4:add(nn.Tanh())
s4:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 128 x fr_w/16 x fr_h/16


s5:add(s4)								--------------------- Conv block 5
s5:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s5:add(nn.Tanh())
s5:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
s5:add(nn.Tanh())
s5:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 128 x fr_w/32 x fr_h/32


s6:add(s5)								--------------------- DeConv block 1
s6:add(nn.SpatialFullConvolution(128, 128, 4, 4, 2, 2, 1, 1))
s6:add(nn.Tanh())
s6:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s6:add(nn.Tanh())									--- ouput_size: 128 x fr_w/16 x fr_h/16

c2:add(s4)
c2:add(s6)											--- ouput_size: (128+128) x fr_w/16 x fr_h/16

s7:add(c2)								--------------------- DeConv block 2
s7:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
s7:add(nn.Tanh())
s7:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s7:add(nn.Tanh())									--- ouput_size: 128 x fr_w/8 x fr_h/8

c3:add(s3)
c3:add(s7)											--- ouput_size: (128+128) x fr_w/16 x fr_h/16

s8:add(c3)								--------------------- DeConv block 3
s8:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
s8:add(nn.Tanh())
s8:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s8:add(nn.Tanh())									--- ouput_size: 128 x fr_w/4 x fr_h/4

c4:add(c1)
c4:add(s8)											--- ouput_size: (192+128) x fr_w/16 x fr_h/16

s9:add(c4)								--------------------- DeConv block 4
s9:add(nn.SpatialFullConvolution(320, 96, 4, 4, 2, 2, 1, 1))
s9:add(nn.Tanh())
s9:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) 
s9:add(nn.Tanh())									--- ouput_size: 96 x fr_w/2 x fr_h/2

s10:add(s9)								--------------------- DeConv block 5
s10:add(nn.SpatialFullConvolution(96, 96, 4, 4, 2, 2, 1, 1))
s10:add(nn.Tanh())
s10:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) 
s10:add(nn.Tanh())									--- ouput_size: 96 x fr_w/1 x fr_h/1

model:add(s10)							--------------------- Output Layer
model:add(nn.SpatialConvolution(96, 3, 3, 3, 1, 1, 1, 1))
model:add(nn.Tanh())									--- ouput_size: 3 x fr_w/1 x fr_h/1
-------------------------------------------------------------------

-- rand_inputs = torch.randn(batch_size, 2, n_channels, frame_height, frame_width)
-- rand_inputs1 = torch.randn(n_channels, frame_height, frame_width)

-- -- these tensors will hold the outer and inner frames of the triplets, respectively
-- input_frames = torch.Tensor(batch_size, 2, n_channels, frame_height, frame_width)
-- output_frames_middle = torch.Tensor(batch_size, n_channels, frame_height, frame_width)

-- -- FIXME: Ensure generalization of this code for more input sizes / parameters
-- -- This will move necessary frames into the input_frames, output_frames_middle tensors
-- for frameNumber=1, batch_size, 1
-- do
-- 	-- print("Frame number in batch: ", frameNumber) -- 1 to 100 inclusive
	
-- 	input_frames[0+frameNumber][1] = trainingFrames[0+frameNumber]
-- 	input_frames[0+frameNumber][2] = trainingFrames[0+frameNumber+2]

-- 	output_frames_middle[0+frameNumber] = trainingFrames[0+frameNumber+1]

-- end


print("Forwarding...")
-- forward random test inputs
-- aa = model:forward(rand_inputs)
-- forward real inputs
-- aa = model:forward(input_frames)
-- print(aa)



-- TRAINING


-- these tensors will hold the outer and inner frames of the triplets, respectively
input_frames = torch.Tensor(2, n_channels, frame_height, frame_width)
output_frames_middle = torch.Tensor(n_channels, frame_height, frame_width)

-- for training iterations
for loopIterator=1, sizeOfTrainingSet-3, 1
do
	print("Training loop:", loopIterator)

	-- get input, output frames (first, last, middle)
	input_frames[1] = trainingFrames[loopIterator] -- first
	input_frames[2] = trainingFrames[loopIterator + 2] -- last
	output_frames_middle = trainingFrames[loopIterator + 1] -- middle

	-- forward data and print loss
	loss = criterion:forward(model:forward(input_frames), output_frames_middle)
	print(loss)

	-- (1) zero the accumulation of the gradients
	model:zeroGradParameters()
	-- (2) accumulate gradients
	model:backward(input_frames, criterion:backward(model.output, output_frames_middle))
	-- (3) update parameters with a varied learning rate
	-- if loss > 0.12 then
	-- 	model:updateParameters(0.20)
	-- elseif loss > 0.075 then
	-- 	model:updateParameters(0.07)
	-- else
	model:updateParameters(0.05)
	-- end
end



-- TESTING

print("Testing...")

-- input_frames = torch.Tensor(batch_size, 2, n_channels, frame_height, frame_width)
-- output_frames_middle = torch.Tensor(batch_size, n_channels, frame_height, frame_width)

for loopIterator=1, sizeOfTestSet-1, 1
do
	-- get input frames, output frame (first, last, middle)
	input_frames[1] = trainingFrames[loopIterator] -- first
	input_frames[2] = trainingFrames[loopIterator + 2] -- last
	
	groundTruthFrame = torch.Tensor(n_channels, frame_height, frame_width)
	groundTruthFrame = testingFrames[loopIterator+1]

	loss = criterion:forward(model:forward(input_frames), groundTruthFrame)
	predictedFrame = model.output


	-- print(output_frames_middle) --1x3x96x160
	-- print(groundTruthFrame) -- 1x3x96x160
	-- print(predictedFrame) --1x3x96x160

	print("Loss:", loss)
	tempPredictedImage = predictedFrame
	tempGroundTruth = groundTruthFrame

	-- -- Scale RGB from [-1..1] to [0..1] to output correctly
	tempPredictedImage = tempPredictedImage + 1
	tempPredictedImage = tempPredictedImage / 2

	tempGroundTruth = tempGroundTruth + 1
	tempGroundTruth = tempGroundTruth / 2


	--[[maxR = 0
	maxG = 0
	maxB = 0
	for loopH=1, 96, 1 do
		for loopW=1, 160, 1 do

			tempR = tempPredictedImage[loopH][loopW]
			tempG = tempPredictedImage[loopH][loopW]
			tempB = tempPredictedImage[3][loopH][loopW]

			if tempR > maxR then
				maxR = tempR
			end			 	
			if tempG > maxG then
				maxG = tempG
			end	
			if tempB > maxB then
				maxB = tempB
			end	
		end
	end

	print("Maxes: ")
	print("Max red:   ",maxR)	
	print("Max green: ",maxG)	
	print("Max blue:  ",maxB)]]--

	filename = "Original" .. tostring(loopIterator) .. ".jpg"
	image.save(filename, tempGroundTruth)

	filename = "Predicted" .. tostring(loopIterator) .. ".jpg"
	image.save(filename, tempPredictedImage)
end




-- -- FIXME: Remove SANITY CHECK
-- predictedImage = torch.Tensor(3,128,128)
-- predictedImage = input_frames[50][2]

-- filename = "test.jpg"
-- image.save(filename, predictedImage)
