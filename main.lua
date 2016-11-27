-- Lance Simmons
-- Fall, 2016

-- DEBUGGING:
local inspect = require 'inspect'


-- REQUIRES:
require('image')
require('nn')
require('optim')
require('math')

load_images = require('load_images')

-- set up the rng for math lib
math.randomseed(os.time())

-- set default tensor type to 32-bit float
torch.setdefaulttensortype('torch.FloatTensor')


-- open frames dir, count frames
f = io.popen('ls frames')
frameCount = 0
for name in f:lines() do 
	-- print(name) 
	frameCount = frameCount + 1
end

print(frameCount, " frames")


-- tensor of all frames in frames directory
allFrames = load_images.load('frames', frameCount)


-- PARAMETERS
numberOfInputImagesToUse = 100

if numberOfInputImagesToUse > allFrames:size(1) then
	print "Bad parameter: numberOfInputImagesToUse > number of frames"
	os.exit()
end



sizeOfTrainingSet = math.floor((9/10) * numberOfInputImagesToUse)
sizeOfTestSet = numberOfInputImagesToUse - sizeOfTrainingSet

print("Training Set Size: ", sizeOfTrainingSet)
print("Test Set Size:     ", sizeOfTestSet)


--  build training set
trainingFrames = 
	torch.Tensor(
	sizeOfTrainingSet,
	allFrames:size(2),
	allFrames:size(3),
	allFrames:size(4))

for frameNumber=1, sizeOfTrainingSet, 1
do
	trainingFrames[frameNumber] = allFrames[frameNumber]:clone()
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
	testingFrames[frameNumber] = allFrames[frameNumber + sizeOfTrainingSet]:clone()
end


os.exit()


