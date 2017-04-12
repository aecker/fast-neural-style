require 'torch'
require 'nn'
require 'image'
require 'camera'

require 'qt'
require 'qttorch'
require 'qtwidget'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'


local cmd = torch.CmdLine()

-- Model options
cmd:option('-models', 'models/models/instance_norm/candy.t7')
cmd:option('-height', 480)
cmd:option('-width', 640)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)

-- Webcam options
cmd:option('-webcam_idx', 0)
cmd:option('-webcam_fps', 60)
cmd:option('-alpha_in', 0.1, 'smoothing parameter for input frames')
cmd:option('-alpha_out', 0.1, 'smoothing parameter for output frames')


local function main()
  local opt = cmd:parse(arg)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local models = {}

  local preprocess_method = nil
  for _, checkpoint_path in ipairs(opt.models:split(',')) do
    print('loading model from ', checkpoint_path)
    local checkpoint = torch.load(checkpoint_path)
    local model = checkpoint.model
    -- insert bilinear upsampling into the model
    local upsamp = nn.SpatialUpSamplingBilinear(2)
    model:insert(upsamp, 1)
    -- model:insert(upsamp, #model + 1)
    model:evaluate()
    model:type(dtype)
    if use_cudnn then
      cudnn.convert(model, cudnn)
    end
    table.insert(models, model)
    local this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
    if not preprocess_method then
      print('got here')
      preprocess_method = this_preprocess_method
      print(preprocess_method)
    else
      if this_preprocess_method ~= preprocess_method then
        error('All models must use the same preprocessing')
      end
    end
  end

  local preprocess = preprocess[preprocess_method]

  local camera_opt = {
    idx = opt.webcam_idx,
    fps = opt.webcam_fps,
    height = opt.height,
    width = opt.width,
  }
  local cam = image.Camera(camera_opt)

  local win = nil
  local img_pre = nil
  local imgs_out = {}
  while true do
    -- Grab a frame from the webcam
    local img = cam:forward()
    img = image.hflip(img)
    -- Preprocess the frame
    local alpha_in = opt.alpha_in
    local H, W = img:size(2), img:size(3)
    img = img:view(1, 3, H, W)
    if img_pre then
      img_pre = (alpha_in * img_pre + (1-alpha_in) * preprocess.preprocess(img):type(dtype))
    else
      img_pre = preprocess.preprocess(img):type(dtype)
    end

    -- Run the models
    for i, model in ipairs(models) do
      --local timer22 = torch.Timer() --Jonas
      local img_out_pre = model:forward(img_pre)
      --if cutorch then cutorch.synchronize() end --Jonas
      --print(timer22:time().real) --Jonas

      -- Deprocess the frame and show the image
      local alpha_out = opt.alpha_out
      local img_out = preprocess.deprocess(img_out_pre)[1]:float()
      if imgs_out[i] then
        imgs_out[i] = (alpha_out * imgs_out[i] + (1-alpha_out) * img_out)
      else
        table.insert(imgs_out, img_out)
      end
    end
    local img_disp = image.toDisplayTensor{
      input = imgs_out,
      min = 0,
      max = 1,
      nrow = math.floor(math.sqrt(#imgs_out)),
    }


    if not win then
      -- On the first call use image.display to construct a window
      win = image.display(img_disp)
    else
      -- Reuse the same window
      win.image = img_out
      local size = win.window.size:totable()
      local qt_img = qt.QImage.fromTensor(img_disp)
      win.painter:image(0, 0, size.width, size.height, qt_img)
    end
  end
end


main()

