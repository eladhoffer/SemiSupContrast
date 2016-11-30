local display = require 'display'
local gnuplot = require 'gnuplot'
local PlotCSV = torch.class('PlotCSV')
function PlotCSV:__init(filename, delim)
  self.filename = filename
  self.delim = delim or ' | '
end

function PlotCSV:parse()
  local function splitFields(s)
    return s:gsub('^%s+',''):gsub('%s+',' '):split(self.delim)
  end
  local f = io.open(self.filename,'r')

  local title = f:read('*l')
  local fields = splitFields(title)

  local data = {}
  for l in f:lines() do
    local values = splitFields(l)
    for i, field in pairs(fields) do
      data[field] = data[field] or {}
      table.insert(data[field], values[i])
    end
  end
  self.data = data
  return self.data
end

function PlotCSV:display(config)
  -- if self.data and config.labels[1] and self.data[config.labels[1]] then
  local plotted = {}
  for i=1, #self.data[config.labels[1]] do
    local entry = {}
    for _,k in pairs(config.labels) do
      table.insert(entry, self.data[k][i])
    end
    table.insert(plotted, entry)
  end
  return display.plot(plotted, config)
  --end
end

function PlotCSV:save(filename, config)
  -- if self.data and config.labels[1] and self.data[config.labels[1]] then
  local plotted = {}
  local x = torch.Tensor(self.data[config.labels[1]])
  local entries = {}
  for i=2, #config.labels do
    local key = config.labels[i]
    table.insert(entries, {key, x, torch.Tensor(self.data[key]),'-'})
  end
  os.execute('rm -f "' .. filename .. '"')
  local epsfig = gnuplot.epsfigure(filename)
  gnuplot.plot(unpack(entries))
  gnuplot.grid('on')
  gnuplot.xlabel(config.labels[1])
  gnuplot.ylabel(config.ylabel)
  gnuplot.title(config.title)
  gnuplot.plotflush()
  gnuplot.close(epsfig)
  --end
end
