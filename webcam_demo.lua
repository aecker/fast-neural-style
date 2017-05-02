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
cmd:option('-cycle', 0)
cmd:option('-cycle_time', 15)

-- Saliency options
cmd:option('-saliency', false, 'if true shows deep gaze predictions instead of stylisation')
cmd:option('-density', false, 'if true the density instead of log-density is displayed')
cmd:option('-imgvis', false, 'if true the saliency map is multiplied with the input image in the output')
cmd:option('-gamma', 1)

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

  if opt.saliency then
    paths.dofile('Misc.lua') -- for LogSoftMax in deepgaze
    opt.models = 'models/models/DeepGaze.t7'
  end

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local models = {}

  local preprocess_method = nil
  for _, checkpoint_path in ipairs(opt.models:split(',')) do
    print('loading model from ', checkpoint_path)
    local model = nil
    local checkpoint = nil
    if opt.saliency then 
      model = torch.load(checkpoint_path)
    else
      checkpoint = torch.load(checkpoint_path)
      model = checkpoint.model
    end
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
    local this_preprocess_method = nil
    if opt.saliency then
      this_preprocess_method = 'vgg'
    else
      this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
    end
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
  local start = os.time()
  local index = 1
  local models2 = {}
  local num_models = #models
  if opt.cycle == 1 then
    table.insert(models2, models[index]:clone())
  else
    models2 = models
  end
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

    if opt.cycle == 1 then
      if (os.time() - start) > opt.cycle_time then
        index = index % num_models + 1
        models2 = {}
        table.insert(models2, models[index]:clone())
        start = os.time()
      end
    end

    -- Run the models
    for i, model in ipairs(models2) do
      --local timer22 = torch.Timer() --Jonas
      local img_out_pre = model:forward(img_pre)
      --if cutorch then cutorch.synchronize() end --Jonas
      --print(timer22:time().real) --Jonas

      -- Deprocess the frame and show the image
      img_out = nil
      if opt.saliency then
        img_out = img_out_pre:float()
	if opt.density then
	  img_out = torch.exp(img_out)
	end
	img_out:add(-img_out:min())
	img_out:div(img_out:max())
	-- img_out:mul(2)
	-- img_out[img_out:ge(1)] = 1
	if opt.imgvis then
	  -- img_out = torch.cmul(img:float(), img_out:expandAs(img))[1]
	  img_out = torch.cmul(img:float() - 0.5, torch.pow(img_out:expandAs(img), opt.gamma))[1] + 0.5
	  -- img_out = torch.cmul(torch.sqrt(img:float()), img_out:expandAs(img))[1]
	else
	  img_out = img_out[1]
	end
      else
        img_out = preprocess.deprocess(img_out_pre)[1]:float()
      end
      local alpha_out = opt.alpha_out
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

