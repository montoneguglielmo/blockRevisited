{
    "configNet":
    {
	"net1":
	{
	    "init": "random",
	    "type": "conv",
	    "params":{"in_channels":12, "out_channels": 6, "kernel_size": [3,3], "stride": 1},
	    "actFun":"rlu",
	    "requires_grad": true
	},
	"net2":
	{
	    "type": "max_pool",
	    "params":{"kernel_size":2}
	},
	"net3":
	{
	    "init": "random",
	    "type": "conv",
	    "params":{"in_channels":6, "out_channels": 16, "kernel_size": [4,4], "stride": 1},
	    "actFun":"rlu",
	    "requires_grad": true
	},
	"net4":
	{
	    "init": "random",
	    "type": "conv",
	    "params":{"in_channels":16, "out_channels": 1, "kernel_size": [4,4], "stride": 1},
	    "actFun":"rlu",
	    "requires_grad": true
	},
	"output0":
	{
	    "init": "random",
	    "type": "fc",
	    "params":{"in_features":3080, "out_features": 5},
	    "actFun":"softmax",
	    "requires_grad": true
	},
        "output1":
	{
	    "init": "random",
	    "type": "fc",
	    "params":{"in_features":3080, "out_features": 5},
	    "actFun":"softmax",
	    "requires_grad": true
	}
    },
    "strctNet":
    {
	"net1"   :{"input":["input"], "output":["net2"]},
	"net2"   :{"input":["net1"], "output":["net3"]},
	"net3"   :{"input":["net2"], "output":["net4"]},
	"net4"   :{"input":["net3"], "output":["output0", "output1"]},
	"output0":{"input":["net4"]},
        "output1":{"input":["net4"]}
    }
}
