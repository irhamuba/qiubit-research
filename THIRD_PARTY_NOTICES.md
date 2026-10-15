# Third-party notices

## stories260K weights, tok512 tokenizer, llama2.c reference code

`weights/stories260K.bin` and `weights/tok512.bin` come from
https://huggingface.co/karpathy/tinyllamas (`stories260K/`). `weights/tensors.json` and
`web/vocab.json` are int8 / JSON re-encodings of these files. The model architecture and the
forward pass in `ref/float_ref.py` follow https://github.com/karpathy/llama2.c.

```
MIT License

Copyright (c) 2023 Andrej

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## TinyStories dataset

The model was trained by its author on TinyStories
(https://huggingface.co/datasets/roneneldan/TinyStories, CDLA-Sharing-1.0; Eldan & Li,
https://arxiv.org/abs/2305.07759). This repository does not redistribute the dataset.

## Octra tools (not included)

`ref/q16.py` re-implements the arithmetic of `lib/vm/machine/fixed_q16.ml` from
https://github.com/octra-labs/amlc (BSD-3-Clause, Copyright (c) 2023-2026 Octra Labs) so the
Python reference matches the VM bit for bit. The `amlc` compiler and the `webcli` wallet
(https://github.com/octra-labs/webcli, GPL-2.0-or-later) are used as external tools and are not
part of this repository.
