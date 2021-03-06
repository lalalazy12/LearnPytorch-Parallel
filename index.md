# Data Parallel


## class DataParallel(Module)
 This container parallelizes the application of the given attr:`module` by splitting the input across the specified devices by chunking in the batch dimension (other objects will be copied once per device). 

```
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        //--------------------1--------------------
        //---------return if gpu availabel--------- 
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
        //--------------------2--------------------
            device_ids = _get_all_device_indices()
        //------Store output in device[0]------
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)
```
1. _get_available_device_type

    ```
    def _get_available_device_type():
        if torch.cuda.is_available():
            return "cuda"
        # add more available device types here
        return None
    ```
    `is_availabel`:
    `return torch._C._cuda_getDeviceCount() > 0`

2. _get_all_device_indices():
    ```
    def _get_all_device_indices():
        # all device index
        return _get_device_attr(lambda m: list(range(m.device_count())))
    ```

## Forward
In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.tensors will be scattered on dim specified (default 0)

```
def forward(self, *inputs, **kwargs):
    with torch.autograd.profiler.record_function("DataParallel.forward"):

        //--------------------check--------------------
        //----check if server(device[0]) has all parameters and buffer---- 
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                     "on device {} (device_ids[0]) but found one of "
                     "them on device: {}".format(self.src_device_obj, t.device))

        //--------------------1--------------------
                  //----scatter inputs---- 
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # for forward function without any inputs, empty list and dict will be created
        # so the module can be executed on one device which is the first one in device_ids
        if not inputs and not kwargs:
            inputs = ((),)
            kwargs = ({},)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        //--------------------2--------------------
                  //----replicate module---- 
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        //--------------------3--------------------
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        //--------------------4--------------------
        return self.gather(outputs, self.output_device)

    //--------------------1--------------------
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    //--------------------2--------------------
    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    //--------------------3--------------------
    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    //--------------------4--------------------
    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

```
**check**:

 Before running, GPU device_ids[0] (that is, our server) must have parallelized module parameters and buffers

### 1.scatter

```
def scatter(self, inputs, kwargs, device_ids):
    return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
```

```
def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs): ???
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
```

Slices tensors into approximately equal chunks and distributes them across given GPUs. Duplicates references to objects that are not tensors.
```
def scatter(inputs, target_gpus, dim=0):
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            //----------scatter input here----------
            return Scatter.apply(target_gpus, None, dim, obj)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res
```

```
class Scatter(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None

        if torch.cuda.is_available() and ctx.input_device == -1:
            #-----Perform CPU to GPU copies in a background stream-----
            streams = [_get_stream(device) for device in target_gpus]
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)
```

### 2.Replicate network


```
def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                        "childrens of ScriptModule")

    if not devices:
        return []

    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    //--------------------1--------------------
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    //----------replicate buffers----------
    buffers = list(network.buffers())
    buffers_rg = []
    buffers_not_rg = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    //----------record buffers(require grad or not)----------
    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    //----------boadcast buffers(like 1)----------
    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}

    //--------------------2--------------------
    //----------replicate modules----------
    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            # This is a temporary fix for DDP. DDP needs to access the
            # replicated model parameters. It used to do so through
            # `mode.parameters()`. The fix added in #33907 for DP stops the
            # `parameters()` API from exposing the replicated parameters.
            # Hence, we add a `_former_parameters` dict here to support DDP. ???
            replica._former_parameters = OrderedDict()

            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [module_copies[j][0] for j in range(num_replicas)]

```


1. 
    ```
    def _broadcast_coalesced_reshape(tensors, devices, detach=False):
        from ._functions import Broadcast
        if detach:
            //--------------------2--------------------
            return comm.broadcast_coalesced(tensors, devices)
        else:
            # Use the autograd function to broadcast if not detach
            if len(tensors) > 0:
                //--------------------1--------------------
                tensor_copies = Broadcast.apply(devices, *tensors)
                return [tensor_copies[i:i + len(tensors)]
                        for i in range(0, len(tensor_copies), len(tensors))]
            else:
                return []
    ```

    1. Boadcast
        ```
        class Broadcast(Function):

            @staticmethod
            def forward(ctx, target_gpus, *inputs):
                assert all(i.device.type != 'cpu' for i in inputs), (
                    'Broadcast function not implemented for CPU tensors'
                )
                target_gpus = [_get_device_index(x, True) for x in target_gpus]
                ctx.target_gpus = target_gpus
                if len(inputs) == 0:
                    return tuple()
                ctx.num_inputs = len(inputs)
                ctx.input_device = inputs[0].get_device()
                //----------same as detach = True----------
                outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
                non_differentiables = []
                for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
                    if not input_requires_grad:
                        for output in outputs:
                            non_differentiables.append(output[idx])
                ctx.mark_non_differentiable(*non_differentiables)
                return tuple([t for tensors in outputs for t in tensors])

            @staticmethod
            def backward(ctx, *grad_outputs):
                return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)
        ```
    2. broadcast_coalesced

        Broadcasts a sequence tensors to the specified GPUs. Small tensors are first coalesced into a buffer to reduce the number of synchronizations.

        Args:
        tensors (sequence): tensors to broadcast. Must be on the same device, either CPU or GPU. 

        devices (Iterable[torch.device, str or int]): an iterable of GPU devices, among which to broadcast. 

        buffer_size (int): maximum size of the buffer used for coalescing

        Returns:

        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.

        ```
        def broadcast_coalesced(tensors, devices, buffer_size=10485760):
            """
            devices = [_get_device_index(d) for d in devices]
            tensors = [_handle_complex(t) for t in tensors]
            return torch._C._broadcast_coalesced(tensors, devices, buffer_size)
        ```
2. Replicate module's network structure, parameters and buffers

### 3.parallel_apply

Both Distributed Paralle and Distributed Data Parallel use this API.

```
    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
```
Applies each `module` in `modules` in parallel on arguments contained in `inputs` and `kwargs_tup` (keyword) on each of `devices`.

Args:
    - modules (Module): modules to be parallelized
    - inputs (tensor): inputs to the modules
    - devices (list of int or torch.device): CUDA devices

`modules`,`inputs`,`kwargs_tup`, and`devices` should all have same length. Moreover, each
element of `inputs` can either be a single object as the only argument
to a module, or a collection of positional arguments.

```
def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
devices = [_get_device_index(x, True) for x in devices]
lock = threading.Lock()
results = {}
grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

def _worker(i, module, input, kwargs, device=None):
    torch.set_grad_enabled(grad_enabled)
    if device is None:
        device = get_a_var(input).get_device()
    try:
        //----------tip 1: what is with torch.cuda.device()----------
        with torch.cuda.device(device), autocast(enabled=autocast_enabled):
            # this also avoids accidental slicing of `input` if it is a Tensor
            if not isinstance(input, (list, tuple)):
                input = (input,)
            //----------in each thread: forward ----------
            output = module(*input, **kwargs)
        with lock:
            results[i] = output
    except Exception:
        with lock:
            results[i] = ExceptionWrapper(
                where="in replica {} on device {}".format(i, device))


if len(modules) > 1:
    //----------init multi thread ----------
    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, device))
                for i, (module, input, kwargs, device) in
                enumerate(zip(modules, inputs, kwargs_tup, devices))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
else:
    _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

outputs = []
for i in range(len(inputs)):
    output = results[i]
    if isinstance(output, ExceptionWrapper):
        output.reraise()
    //----------get all result ----------
    outputs.append(output)
return outputs
```
[tip 1](https://stackoverflow.com/questions/52076815/pytorch-use-device-inside-with-statement): 

it can just be used to switch between cuda devices, not copy data from one device to another device (e.g. copy from cpu to gpu ). And [method](https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu) to get current device.

### 4.Gather outputs
Gather all outputs to device[0]






## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/lalalazy12/LearnPytorch-Parallel/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/lalalazy12/LearnPytorch-Parallel/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we???ll help you sort it out.
