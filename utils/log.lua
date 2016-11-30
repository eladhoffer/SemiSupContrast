local argcheck = require 'argcheck'
local tnt = require 'torchnet'
local logtext = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'

getLog = argcheck{
    {name="logFile", type="string"},
    {name="keys", type="table"},
    {name="nestFormat", type="string", default="%s (Top %s)"},
    call = function(logFile, keys, nestFormat)
        local logKeys = {}
        for id, key in pairs(keys) do
            if type(key) == 'table' then
                for _, subKey in pairs(key) do
                    table.insert(logKeys, (nestFormat):format(id, subKey))
                end
            else
                table.insert(logKeys, key)
            end
        end

        local logKeysFormat = {}
        local logFormat = {}
        for i in pairs(logKeys) do
            table.insert(logKeysFormat, '%10.3f')
            table.insert(logFormat, '%10s')
        end

        local log = tnt.Log{
            keys = logKeys,
            onFlush = {
                -- write out all keys in "log" file
                logtext{filename=logFile, keys=logKeys, format=logKeysFormat},
                logtext{keys=logKeys},
                -- addPlot('Classification Error', {'Epoch','Train Error (Top 1)', 'Test Error (Top 1)'}, 'Error %'),
            },
            onSet = {
                -- add status to log
                logstatus{filename=logFile}
            }
        }

        log:set{
            __status__ = string.format(table.concat(logFormat, ' | '), unpack(logKeys)),
        }

        log.set = argcheck{
            {name="self", type="tnt.Log"},
            {name="keys", type="table"},
            call =
            function(self, keys)
                for key, value in pairs(keys) do
                    if type(value) == 'table' then
                        for subKey, subVal in pairs(value) do
                            local newKey = (nestFormat):format(key, subKey)
                            tnt.Log.set(self, {[newKey] = subVal})
                        end
                    else
                        tnt.Log.set(self, {[key] = value})
                    end
                end
            end
        }

        return log
    end
}

function updateOpt(optState, epoch, regime, verbose)
    if regime and regime.epoch then
        for epochNum, epochVal in pairs(regime.epoch) do
            if epochVal == epoch then
                for optValue,_ in pairs(regime) do
                    if regime[optValue][epochNum] then
                        if verbose then
                            print(optValue,': ',optState[optValue], ' -> ', regime[optValue][epochNum])
                        end
                        optState[optValue] = regime[optValue][epochNum]
                    end
                end
            end
        end
    end
end

function clonedSavedModel(model)
    local savedModel = model:clone('weight', 'bias', 'running_mean', 'running_std', 'running_var')
    savedModel:apply(function(m)
        if m.gradWeight then
            m.gradWeight = m.gradWeight.new()
        end
        if m.gradBias then
            m.gradBias = m.gradBias.new()
        end

    end
    )
    return savedModel
end
function inflateGradModel(model)
    model:apply(function(m)
        if m.gradWeight then
            m.gradWeight:resizeAs(m.weight)
        end
        if m.gradBias then
            m.gradBias:resizeAs(m.bias)
        end
    end
    )
    return model
end
