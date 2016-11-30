require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'data'
require 'utils.log'
require 'utils.plotCSV'
require 'ContrastingMinEntropyCriterion'
local tnt = require 'torchnet'
----------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder', './models/', 'Models Folder')
cmd:option('-model', 'MNIST_contrast.lua', 'Model file - must return valid network.')
cmd:option('-LR', 0.1, 'learning rate')
cmd:option('-LRDecay', 0, 'learning rate decay (in # samples)')
cmd:option('-weightDecay', 1e-4, 'L2 penalty on the weights')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-batchSize', 32, 'batch size')
cmd:option('-optimization', 'sgd', 'optimization method')
cmd:option('-epoch', -1, 'number of epochs to train, -1 for unbounded')
cmd:option('-crossContrast', true, '')
cmd:option('-numLabeled', 10, '')
cmd:option('-numCompared', 1, '')
cmd:option('-testOnly', false, '')
cmd:option('-saveFeats', false, '')

cmd:text('===>Platform Optimization')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-seed', 1000, 'seed number')
cmd:option('-type', 'cuda', 'cuda/cl/float/double')
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-nGPU', 1, 'num of gpu devices used')
cmd:option('-saveOptState', false, 'Save optimization state every epoch')

cmd:text('===>Save/Load Options')
cmd:option('-load', '', 'load existing net')
cmd:option('-resume', false, 'resume training from the same epoch')
cmd:option('-save', os.date():gsub(' ',''), 'name of saved directory')

cmd:text('===>Data Options')
cmd:option('-dataset', 'MNIST', 'Dataset - ImageNet, Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-augment', false, 'Augment training data')

opt = cmd:parse(arg or {})
opt.model = opt.modelsFolder .. paths.basename(opt.model, '.lua')
opt.savePath = paths.concat('./results', opt.save)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Output files configuration
os.execute('mkdir -p ' .. opt.savePath)
cmd:log(opt.savePath .. '/Log.txt', opt)
local logFile = paths.concat(opt.savePath, 'LogTable.csv')


local log = getLog{
  logFile = logFile,
  keys = {'Epoch','Train Loss','Test Loss', 'Train Error', 'Test Error'}
}

local plots = {
  {
    title = opt.save .. ':Loss',
    labels = {'Epoch', 'Train Loss', 'Test Loss'},
    ylabel = 'Loss'
  },
  {
    title = ('%s : Classification Error'):format(opt.save),
    labels = {'Epoch', 'Train Error', 'Test Error'},
    ylabel = 'Error %'
  }
}

log:attach('onFlush',
  {
    function()
      local plot = PlotCSV(logFile)
      plot:parse()
      for _,p in pairs(plots) do
        if pcall(require , 'display') then
          p.win = plot:display(p)
        end
        plot:save(paths.concat(opt.savePath, p.title:gsub('%s','') .. '.eps'), p)
      end
    end
  }
)

local netFilename = paths.concat(opt.savePath, 'savedModel')

----------------------------------------------------------------------
local config = {
  inputSize = 32,
  reshapeSize = 32,
  inputMean = 128,
  inputStd = 128,
  regime = {}
}
local function setConfig(target, origin, overwrite)
  for key in pairs(config) do
    if overwrite or target[key] == nil then
      target[key] = origin[key]
    end
  end
end

-- Model + Loss:
local model, criterion
if paths.filep(opt.load) then
  local conf, criterion = require(opt.model)
  model = torch.load(opt.load)
  if not opt.resume then
    setConfig(model, conf)
  end
else
  model, criterion = require(opt.model)
end

criterion = criterion or nn.ContrastingMinEntropyCriterion(2)

setConfig(model, config)

-- Data preparation
local trainData = getDataset(opt.dataset, 'train')
local testData = getDataset(opt.dataset, 'test')
-- classes
local classes = trainData:classes()
local numClasses = #classes

local evalTransform = tnt.transform.compose{
  Scale(model.reshapeSize),
  CenterCrop(model.inputSize),
  Normalize(model.inputMean, model.inputStd)
}

local augTransform = tnt.transform.compose{
  RandomCrop(model.inputSize, 4),
  HorizontalFlip(),
  Normalize(model.inputMean,model.inputStd),
}

testData = tnt.TransformDataset{
  transforms = {
    input = evalTransform
  },
  dataset = testData
}

trainData = tnt.TransformDataset{
  transforms = {
    input = (opt.augment and augTransform) or evalTransform
  },
  dataset = trainData:shuffle()
}

local labeledData = extractEachTarget{dataset=trainData,num=opt.numLabeled}
local trainLabeledData = tnt.ConcatDataset{datasets=labeledData}
local labeledIter = getIterator(trainLabeledData:batch(opt.batchSize), opt.threads, opt.seed)
local trainIter = getIterator(trainData:batch(opt.batchSize), opt.threads, opt.seed)
local testIter = getIterator(testData:batch(opt.batchSize), opt.threads, opt.seed)

----------------------------------------------------------------------
-- Model optimization

local types = {
  cuda = 'torch.CudaTensor',
  float = 'torch.FloatTensor',
  cl = 'torch.ClTensor',
  double = 'torch.DoubleTensor'
}

local tensorType = types[opt.type] or 'torch.FloatTensor'

if opt.type == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.devid)
  cutorch.manualSeed(opt.seed)
  local cudnnAvailable = pcall(require , 'cudnn')
  if cudnnAvailable then
    cudnn.benchmark = true
    model = cudnn.convert(model, cudnn)
  end
elseif opt.type == 'cl' then
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.devid)
end

model:type(tensorType)
criterion = criterion:type(tensorType)

---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
  local net = model
  model = nn.DataParallelTable(1, true, true)
  local useCudnn = cudnn ~= nil
  local modelConf = opt.model
  model:add(net, torch.range(1, opt.nGPU):totable()) -- Use the ith GPU
  model:threads(function()
      require(modelConf)
      if useCudnn then
        require 'cudnn'
        cudnn.benchmark = true
      end
    end
  )
  setConfig(model, net, true)
end

-- Optimization configuration
local weights,gradients = model:getParameters()
local savedModel = model
if opt.nGPU > 1 then
  savedModel = savedModel:get(1)
end
savedModel = savedModel:clone('weight', 'bias', 'running_mean', 'running_std', 'running_var')

------------------Optimization Configuration--------------------------
local optimState = model.optimState or {
  method = opt.optimization,
  learningRate = opt.LR,
  momentum = opt.momentum,
  dampening = 0,
  weightDecay = opt.weightDecay,
  learningRateDecay = opt.LRDecay
}

----------------------------------------------------------------------
print '==> Network'
print(model)
print('==>' .. weights:nElement() .. ' Parameters')

print '==> Criterion'
print(criterion)

----------------------------------------------------------------------
if opt.savePathOptState then
  model.optimState = optimState
end

local epoch = (model.epoch and model.epoch + 1) or 1

local function forward(dataIterator, train)
  local x = torch.Tensor():type(tensorType)
  local dE_dy = torch.Tensor():type(tensorType)
  local sizeData = dataIterator:execSingle('fullSize')
  local numSamples = 0
  local lossMeter = tnt.AverageValueMeter()
  lossMeter:reset();

  local contrast = {}
  for i=1, numClasses do
    contrast[i] = labeledData[i]:shuffle()
  end
  local numContrast = numClasses * (opt.numCompared + 1)
  local yt = torch.Tensor(numClasses):type(tensorType)
  for sample in dataIterator() do
    local inputSize = sample.input:size()
    local numUnlabeled = sample.input:size(1)
    inputSize[1] = numUnlabeled + numContrast
    x:resize(inputSize)
    --copy unlabeled examples
    x:sub(1, numUnlabeled):copy(sample.input)
    --copy compared labeled data
    for i=1, numContrast do
      local c = (i-1) % numClasses + 1
      local idx = torch.random(opt.numLabeled)
      x[numUnlabeled + i]:copy(contrast[c]:get(idx).input)
    end
    local y = model:forward(x)
    local yL = y:sub(1, numUnlabeled + numClasses)
    local yC = y:sub(-numClasses*opt.numCompared, -1)

    yt:resize(numUnlabeled+ numContrast):zero()
    -- yt:copy(-numClasses*opt.numCompared, -1):copy(torch.range(1,numClasses))
    yt:sub(-numClasses*opt.numCompared, -1):copy(torch.range(1,numClasses))
    local loss, assignment = criterion:forward({yL, yC},yt)
    if train then
      local function feval()
        model:zeroGradParameters()
        local dE_dyCL = criterion:backward({yL, yC}, yt)
        dE_dy:resizeAs(model.output)
        dE_dy:sub(1, numUnlabeled + numClasses):copy(dE_dyCL[1])
        dE_dy:sub(-numClasses*opt.numCompared, -1):copy(dE_dyCL[2])
        model:backward(x, dE_dy)
        return loss, gradients
      end
      _G.optim[optimState.method](feval, weights, optimState)
      if opt.nGPU > 1 then
        model:syncParameters()
      end
    end
    if torch.type(y) == 'table' then y = y[1] end
    lossMeter:add(loss, sample.input:size(1) / opt.batchSize)
    numSamples = numSamples + sample.input:size(1)
    xlua.progress(numSamples, sizeData)
  end
  return lossMeter:value()
end

------------------------------
local function train(dataIterator)
  model:training()
  return forward(dataIterator, true)
end

local function test(dataIterator)
  model:evaluate()
  return forward(dataIterator, false)
end
------------------------------

local function extractFeats(dataIterator, sizeData)
  local Feats = torch.Tensor():type(tensorType)
  local Labels = torch.Tensor():type(tensorType)
  local first = true
  local x = torch.Tensor():type(tensorType)
  local num = 1
  for sample in dataIterator() do
    x:resize(sample.input:size()):copy(sample.input)
    local y = model:forward(x)
    if first then
      Feats:resize(sizeData, y:size(2))
      Labels:resize(sizeData)
    end
    Feats:narrow(1,num,x:size(1)):copy(y)
    Labels:narrow(1,num,x:size(1)):copy(sample.target)
    num = num + x:size(1)
  end
  return Feats, Labels
end

-- function that performs nearest neighbor classification:
local function nn_classification(train_Z, train_Y, test_Z)

  -- compute squared Euclidean distance matrix between train and test data:
  local N = train_Z:size(1)
  local M = test_Z:size(1)
  local buff1 = train_Z.new():resize(train_Z:size())
  local buff2 = test_Z.new():resize(test_Z:size())
  torch.cmul(buff1, train_Z, train_Z)
  torch.cmul(buff2, test_Z, test_Z)
  local sum_Z1 = buff1:sum(2)
  local sum_Z2 = buff2:sum(2)
  local sum_Z1_expand = sum_Z1:t():expand(M, N)
  local sum_Z2_expand = sum_Z2:expand(M, N)
  local D = torch.mm(test_Z, train_Z:t())
  D:mul(-2)
  D:add(sum_Z1_expand):add(sum_Z2_expand)

  -- perform 1-nearest neighbor classification:
  local test_Y = train_Y.new():resize(M)
  for m = 1, M do
    local _,ind = torch.min(D[m], 1)
    test_Y[m] = train_Y[ind[1]]
  end

  -- return classification
  return test_Y
end

local function test_nn(trainDataIterator, testDataIterator)
  model:evaluate()
  local trainFeats, trainLabels = extractFeats(trainDataIterator, trainLabeledData:size())
  local testFeats, testLabels = extractFeats(testDataIterator, testData:size())
  if opt.saveFeats then
    torch.save(paths.concat(opt.savePath, 'trainFeats_epoch' .. epoch), {trainFeats:float(), trainLabels:float()})
    torch.save(paths.concat(opt.savePath, 'testFeats_epoch' .. epoch), {testFeats:float(), testLabels:float()})
  end
  testPred = nn_classification(trainFeats, trainLabels, testFeats)
  return testPred:ne(testLabels):float():mean() * 100
end

local lowestTestError = 100
print '\n==> Starting Training\n'

while epoch ~= opt.epoch do
  if not opt.testOnly then
    model.epoch = epoch
    log:set{Epoch = epoch}
    print('\nEpoch ' .. epoch)
    updateOpt(optimState, epoch, model.regime, true)
    print('Training:')
    --Train
    trainIter:exec('manualSeed', epoch)
    trainIter:exec('resample')
    local trainLoss = train(trainIter)
    log:set{['Train Loss'] = trainLoss, ['Train Error'] = 0}
    torch.save(netFilename, savedModel)
  end
  --Test
  print('Test:')
  local testLoss = test(testIter)
  local testClassError = test_nn(labeledIter, testIter)
  log:set{['Test Loss'] = testLoss, ['Test Error'] = testClassError}
  print('Nearest Neighbor Error: ' .. testClassError)

  if opt.testOnly then
    break
  end
  log:flush()

  if lowestTestError > testClassError then
    lowestTestError = testClassError
    os.execute(('cp %s %s'):format(netFilename, netFilename .. '_best'))
  end

  epoch = epoch + 1
end
