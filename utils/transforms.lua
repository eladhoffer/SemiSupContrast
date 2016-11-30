--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
require 'image'
function Compose(transforms)
    return function(input)
        for _, transform in ipairs(transforms) do
            input = transform(input)
        end
        return input
    end
end

function Normalize(mean, std)
    return function(x)
        return x:float():add(-mean):div(std)
    end
end

-- Scales the smaller edge to size
function Scale(size, interpolation)
    interpolation = interpolation or 'bicubic'
    return function(input)
        local w, h = input:size(3), input:size(2)
        if (w <= h and w == size) or (h <= w and h == size) then
            return input
        end
        if w < h then
            return image.scale(input, size, h/w * size, interpolation)
        else
            return image.scale(input, w/h * size, size, interpolation)
        end
    end
end

-- Crop to centered rectangle
function CenterCrop(size)
    return function(input)
        local w1 = math.ceil((input:size(3) - size)/2)
        local h1 = math.ceil((input:size(2) - size)/2)
        return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
    end
end

-- Random crop form larger image with optional zero padding
function RandomCrop(size, padding)
    padding = padding or 0

    return function(input)
        if padding > 0 then
            local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
            temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
            input = temp
        end

        local w, h = input:size(3), input:size(2)
        if w == size and h == size then
            return input
        end

        local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
        local out = image.crop(input, x1, y1, x1 + size, y1 + size)
        assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
        return out
    end
end

-- Four corner patches and center crop from image and its horizontal reflection
function TenCrop(size)
    local centerCrop = CenterCrop(size)

    return function(input)
        local w, h = input:size(3), input:size(2)

        local output = {}
        for _, img in ipairs{input, image.hflip(input)} do
            table.insert(output, centerCrop(img))
            table.insert(output, image.crop(img, 0, 0, size, size))
            table.insert(output, image.crop(img, w-size, 0, w, size))
            table.insert(output, image.crop(img, 0, h-size, size, h))
            table.insert(output, image.crop(img, w-size, h-size, w, h))
        end

        -- View as mini-batch
        for i, img in ipairs(output) do
            output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
        end

        return input.cat(output, 1)
    end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function RandomScale(minSize, maxSize)
    return function(input)
        local w, h = input:size(3), input:size(2)

        local targetSz = torch.random(minSize, maxSize)
        local targetW, targetH = targetSz, targetSz
        if w < h then
            targetH = torch.round(h / w * targetW)
        else
            targetW = torch.round(w / h * targetH)
        end

        return image.scale(input, targetW, targetH, 'bicubic')
    end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function RandomSizedCrop(size)
    local scale = Scale(size)
    local crop = CenterCrop(size)

    return function(input)
        local attempt = 0
        repeat
            local area = input:size(2) * input:size(3)
            local targetArea = torch.uniform(0.08, 1.0) * area

            local aspectRatio = torch.uniform(3/4, 4/3)
            local w = torch.round(math.sqrt(targetArea * aspectRatio))
            local h = torch.round(math.sqrt(targetArea / aspectRatio))

            if torch.uniform() < 0.5 then
                w, h = h, w
            end

            if h <= input:size(2) and w <= input:size(3) then
                local y1 = torch.random(0, input:size(2) - h)
                local x1 = torch.random(0, input:size(3) - w)

                local out = image.crop(input, x1, y1, x1 + w, y1 + h)
                assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

                return image.scale(out, size, size, 'bicubic')
            end
            attempt = attempt + 1
        until attempt >= 10

        -- fallback
        return crop(scale(input))
    end
end

function HorizontalFlip(prob)
    local prob = prob or 0.5
    return function(input)
        if torch.uniform() < prob then
            input = image.hflip(input)
        end
        return input
    end
end

function Rotation(deg)
    return function(input)
        if deg ~= 0 then
            input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
        end
        return input
    end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function Lighting(alphastd, eigval, eigvec)
    local alphastd = alphastd or 0.1
    local eigval = eigval or torch.Tensor{ 0.2175, 0.0188, 0.0045 }
    local eigvec = eigvecor or torch.Tensor{
        { -0.5675,  0.7192,  0.4009 },
        { -0.5808, -0.0045, -0.8140 },
        { -0.5836, -0.6948,  0.4203 }
    }

    return function(input)
        if alphastd == 0 then
            return input
        end

        local alpha = torch.Tensor(3):normal(0, alphastd)
        local rgb = eigvec:clone()
        :cmul(alpha:view(1, 3):expand(3, 3))
        :cmul(eigval:view(1, 3):expand(3, 3))
        :sum(2)
        :squeeze()

        input = input:clone()
        for i=1,3 do
            input[i]:add(rgb[i])
        end
        return input
    end
end

local function blend(img1, img2, alpha)
    return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
    dst:resizeAs(img)
    dst[1]:zero()
    dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
    dst[2]:copy(dst[1])
    dst[3]:copy(dst[1])
    return dst
end

function Saturation(var)
    local gs

    return function(input)
        gs = gs or input.new()
        grayscale(gs, input)

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(input, gs, alpha)
        return input
    end
end

function Brightness(var)
    local gs

    return function(input)
        gs = gs or input.new()
        gs:resizeAs(input):zero()

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(input, gs, alpha)
        return input
    end
end

function Contrast(var)
    local gs

    return function(input)
        gs = gs or input.new()
        grayscale(gs, input)
        gs:fill(gs[1]:mean())

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(input, gs, alpha)
        return input
    end
end

function RandomOrder(ts)
    return function(input)
        local img = input.img or input
        local order = torch.randperm(#ts)
        for i=1,#ts do
            img = ts[order[i]](img)
        end
        return input
    end
end

function ColorJitter(brightness, contrast, saturation)
    local brightness = brightness or 0
    local contrast = contrast or 0
    local saturation = saturation or 0

    local ts = {}
    if brightness ~= 0 then
        table.insert(ts, Brightness(brightness))
    end
    if contrast ~= 0 then
        table.insert(ts, Contrast(contrast))
    end
    if saturation ~= 0 then
        table.insert(ts, Saturation(saturation))
    end

    if #ts == 0 then
        return function(input) return input end
    end

    return RandomOrder(ts)
end


function GlimpseSeries(minSize, outSize, numPatches)
    local minSize = minSize or 64
    local outSize = outSize or 64
    local numPatches = numPatches or 16
    local genImage = function(img)
        local function getRange(loc, sz, min, max)
            local half = math.ceil(sz / 2)
            local start = math.max(min, loc - half)


            local finish = start + sz - 1
            if finish > max then
                start = max - sz + 1
                finish = max
            end
            assert(start >= min, 'wrong size')
            return start, finish
        end

        local c, h, w = unpack(img:size():totable())
        local maxSize = math.min(w,h)

        local count = torch.ByteTensor(h,w):zero()
        local prob = torch.FloatTensor(h,w)
        local output = torch.FloatTensor(numPatches, c, outSize, outSize)
        local nextLoc = math.ceil((h+1)*w/2)
        local length = maxSize

        for i=1, numPatches do
            local y = math.ceil(nextLoc / w)
            local x = w - nextLoc % w

            local x1, x2 = getRange(x, length, 1, w)
            local y1, y2 = getRange(y, length, 1, h)
            count[{{y1,y2},{x1,x2}}]:add(1)
            local crop = image.crop(img, x1 - 1, y1 - 1, x2 - 1, y2 - 1)
            local scaled = image.scale(crop, outSize, outSize)--, 'bicubic')
            output[i]:copy(scaled)

            prob:copy(count):mul(-1):exp()
            prob:div(prob:sum())

            nextLoc = torch.multinomial(prob:view(-1), 1)[1]
            length = torch.random(minSize, maxSize)
        end

        return output
    end
    return genImage
end

function AddGridMaps()
  local function gridMap(h,w)
    local x = torch.FloatTensor(2,h,w)
    x[{1,{},{}}]:copy(torch.linspace(-1,1,h):view(h,1):expand(h,w))
    x[{2,{},{}}]:copy(torch.linspace(-1,1,w):view(1,w):expand(h,w))
    return x
  end
  return function(x)
    return torch.cat(x, gridMap(x:size(2), x:size(3)):typeAs(x),1)
  end
end
