# Data Parallel

## class DataParallel(Module)

### __init__

 This container parallelizes the application of the given attr:`module` by splitting the input across the specified devices by chunking in the batch dimension (other objects will be copied once per device). In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.tensors will be scattered on dim specified (default 0)

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
    # all device index
    return _get_device_attr(lambda m: list(range(m.device_count()))) 
    get a device list
    ```
    def _get_all_device_indices():
        # all device index
        return _get_device_attr(lambda m: list(range(m.device_count())))
    ```

### Forward

```
def forward(self, *inputs, **kwargs):
    with torch.autograd.profiler.record_function("DataParallel.forward"):

        //--------------------1--------------------
        //----check if server(device[0]) has all parameters and buffer---- 
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                    "on device {} (device_ids[0]) but found one of "
                                    "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # for forward function without any inputs, empty list and dict will be created
        # so the module can be executed on one device which is the first one in device_ids
        if not inputs and not kwargs:
            inputs = ((),)
            kwargs = ({},)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
```


Before running, GPU device_ids[0] (that is, our server) must have parallelized module parameters and buffers




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

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
