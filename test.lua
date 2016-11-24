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

os.exit()


