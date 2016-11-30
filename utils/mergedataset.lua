--[[
Copyright (c) 2016-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local transform = require 'torchnet.transform'

local MergeDataset =
torch.class('tnt.MergeDataset', 'tnt.Dataset', tnt)

MergeDataset.__init = argcheck{
  doc = [[
  <a name="MergeDataset">
  #### tnt.MergeDataset(@ARGP)
  @ARGT

  Given a Lua array (`datasets`) of [tnt.Dataset](#Dataset), concatenates
  them into a single dataset. Each entry will be concatenation of entries from all datasets.

  Purpose: useful to assemble different existing datasets
  ]],
  noordered=true,
  {name='self', type='tnt.MergeDataset'},
  {name='datasets', type='table'},
  {name='merge', type='function', opt=true},
  call =
  function(self, datasets, merge)
    assert(#datasets > 0, 'datasets should not be an empty table')
    local size
    for i, dataset in ipairs(datasets) do
      size = size or dataset:size()
      assert(torch.isTypeOf(dataset, 'tnt.Dataset'),
      'each member of datasets table should be a tnt.Dataset')
      assert(size == dataset:size(),
      'each member of datasets table should have the same size')
    end
    self.makebatch = transform.makebatch{merge=merge}
    self.__datasets = datasets
    self.__size = size
  end
}

MergeDataset.size = argcheck{
  {name='self', type='tnt.MergeDataset'},
  call =
  function(self)
    return self.__size
  end
}

MergeDataset.get = argcheck{
  {name='self', type='tnt.MergeDataset'},
  {name='idx', type='number'},
  call =
  function(self, idx)
    local samples = {}
    for i, dataset in ipairs(self.__datasets) do
      table.insert(samples, dataset:get(idx))
    end
    samples = self.makebatch(samples)
    collectgarbage()
    collectgarbage()
    return samples
  end
}
