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
batch_size = 100
frame_width = 160
frame_height = 96
n_channels = 3



-- CODE START:

-- set default tensor type to 32-bit float
torch.setdefaulttensortype('torch.FloatTensor')

-- set up the rng for math lib
math.randomseed(os.time())

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

if ((numberOfInputFramesToUse % frameGroupSize) ~= 0) then
	print "Bad parameter: numberOfInputFramesToUse % frameGroupSize ~= 0"
	os.exit()
end

if ((numberOfInputFramesToUse % 10) ~= 0) then
	print "Bad parameter: numberOfInputFramesToUse % 10 ~= 0"
	os.exit()
end




sizeOfTrainingSet = math.floor((9/10) * numberOfInputFramesToUse)
sizeOfTestSet = numberOfInputFramesToUse - sizeOfTrainingSet

print("Training Set Size: ", sizeOfTrainingSet)
print("Test Set Size:     ", sizeOfTestSet)

os.exit()


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











input_frames = torch.Tensor(batch_size, 2, n_channels, frame_height, frame_width)
output_frame2 = torch.Tensor(batch_size, n_channels, frame_height, frame_width)

------------ Model ------------------------------------------------

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
c1 = nn.Parallel(2, 1)
c2 = nn.DepthConcat(1)
c3 = nn.DepthConcat(1)
c4 = nn.DepthConcat(1)

s1:add(nn.SpatialConvolution(3, 96, 3, 3, 1, 1, 1, 1)) ----- frame 1 conv block 1
s1:add(nn.ReLU())
s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s1:add(nn.ReLU())
s1:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/2 x fr_h/2

s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) ------ frame 1 conv block 2
s1:add(nn.ReLU())
s1:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s1:add(nn.ReLU())	
s1:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/4 x fr_h/4

s2:add(nn.SpatialConvolution(3, 96, 3, 3, 1, 1, 1, 1)) ----- frame 3 conv block 1
s2:add(nn.ReLU())
s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s2:add(nn.ReLU())
s2:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/2 x fr_h/2

s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) ----- frame 3 conv block 2
s2:add(nn.ReLU())
s2:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1))
s2:add(nn.ReLU())
s2:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 96 x fr_w/4 x fr_h/4

c1:add(s1)
c1:add(s2)											--- ouput_size: (96+96) x fr_w/4 x fr_h/4


s3:add(c1)								--------------------- Conv block 3
s3:add(nn.SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1)) 
s3:add(nn.ReLU())
s3:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
s3:add(nn.ReLU())
s3:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 128 x fr_w/8 x fr_h/8


s4:add(s3)								--------------------- Conv block 4
s4:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s4:add(nn.ReLU())
s4:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
s4:add(nn.ReLU())
s4:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 128 x fr_w/16 x fr_h/16


s5:add(s4)								--------------------- Conv block 5
s5:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s5:add(nn.ReLU())
s5:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
s5:add(nn.ReLU())
s5:add(nn.SpatialMaxPooling(2, 2, 2, 2))		--- ouput_size: 128 x fr_w/32 x fr_h/32


s6:add(s5)								--------------------- DeConv block 1
s6:add(nn.SpatialFullConvolution(128, 128, 4, 4, 2, 2, 1, 1))
s6:add(nn.ReLU())
s6:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s6:add(nn.ReLU())									--- ouput_size: 128 x fr_w/16 x fr_h/16

c2:add(s4)
c2:add(s6)											--- ouput_size: (128+128) x fr_w/16 x fr_h/16

s7:add(c2)								--------------------- DeConv block 2
s7:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
s7:add(nn.ReLU())
s7:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s7:add(nn.ReLU())									--- ouput_size: 128 x fr_w/8 x fr_h/8

c3:add(s3)
c3:add(s7)											--- ouput_size: (128+128) x fr_w/16 x fr_h/16

s8:add(c3)								--------------------- DeConv block 3
s8:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
s8:add(nn.ReLU())
s8:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)) 
s8:add(nn.ReLU())									--- ouput_size: 128 x fr_w/4 x fr_h/4

c4:add(c1)
c4:add(s8)											--- ouput_size: (192+128) x fr_w/16 x fr_h/16

s9:add(c4)								--------------------- DeConv block 4
s9:add(nn.SpatialFullConvolution(320, 96, 4, 4, 2, 2, 1, 1))
s9:add(nn.ReLU())
s9:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) 
s9:add(nn.ReLU())									--- ouput_size: 96 x fr_w/2 x fr_h/2

s10:add(s9)								--------------------- DeConv block 5
s10:add(nn.SpatialFullConvolution(96, 96, 4, 4, 2, 2, 1, 1))
s10:add(nn.ReLU())
s10:add(nn.SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1)) 
s10:add(nn.ReLU())									--- ouput_size: 96 x fr_w/1 x fr_h/1

model:add(s10)							--------------------- Output Layer
model:add(nn.SpatialConvolution(96, 3, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())									--- ouput_size: 3 x fr_w/1 x fr_h/1
-------------------------------------------------------------------


inputs = torch.randn(batch_size, 2, n_channels, frame_height, frame_width)
inputs1 = torch.randn(n_channels, frame_height, frame_width)

aa = model:forward(inputs)
print(aa)

