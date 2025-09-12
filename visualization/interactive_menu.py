
import ipywidgets as widgets
from IPython.display import display
import io
import albumentations as A

import numpy as np
import torch
from PIL import Image
import cv2

class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        # torchvision datasets give PIL.Image, convert to numpy
        img = np.array(img)
        augmented = self.aug(image=img)
        img = augmented["image"]
        # convert back to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.ndim == 2:  # add channel dim if grayscale
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[2] in [1,3]:
                img = img.permute(2,0,1)  # HWC -> CHW
        return img

class AugVisualizer:
    def __init__(self):
        self.rotate = 0
        self.translate_x=0.1
        self.translate_y=0.1
        self.scale=0.1
        self.gaussian_blur=False
        self.gaussian_kernel_size=3
        self.blur_sigma=0.5, 
        self.horizontal_flip=False, 
        self.vertical_flip=False
        self.random_brightness_contrast=False
        self.rgb_shift=False
        
    def transform_contrail(self, rotate=0, translate_x=0.1, translate_y=0.1, scale=0.1, 
                           gaussian_blur=False, gaussian_kernel_size=3, blur_sigma=0.5, 
                           horizontal_flip=False, vertical_flip=False, 
                           random_brightness_contrast=False, rgb_shift=False, pad_mode=0):
        
        contrail_img = np.array(Image.open("contrail-seg/data/goes/florida/image/florida_2020_03_05_0101.png").convert("RGB"))
        contrail_label = np.array(Image.open("contrail-seg/data/goes/florida/mask/florida_2020_03_05_0101.png"))
        
            # Determine padding mode
        border_mode = 0
        if pad_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT_101
        elif pad_mode == "replicate":
            border_mode = cv2.BORDER_REPLICATE

        # Albumentations pipeline
        aug = A.Compose([
            A.Rotate(limit=(rotate, rotate), border_mode=border_mode, p=1.0),
            A.Affine(translate_percent={'x': translate_x, 'y': translate_y}, border_mode=border_mode, scale=(scale, scale), p=1.0),
            A.GaussianBlur(blur_limit=gaussian_kernel_size, sigma_limit=(blur_sigma, blur_sigma), p=1.0 if gaussian_blur else 0.0),
            A.HorizontalFlip(p=1.0 if horizontal_flip else 0.0),
            A.VerticalFlip(p=1.0 if vertical_flip else 0.0),
            A.RandomBrightnessContrast(p=1.0 if random_brightness_contrast else 0.0),
            A.RGBShift(p=1.0 if rgb_shift else 0.0),
            
            
        ])

        # Apply to both image and mask
        augmented = aug(image=contrail_img, mask=contrail_label)
        aug_img = Image.fromarray(augmented["image"])
        aug_label = Image.fromarray(augmented["mask"])

        # Helper for widget conversion
        def to_widget(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return widgets.Image(value=buf.getvalue(), format='png', width=300, height=300)

        # Create widgets
        img_widget = to_widget(Image.fromarray(contrail_img))
        label_widget = to_widget(Image.fromarray(contrail_label))
        aug_img_widget = to_widget(aug_img)
        aug_label_widget = to_widget(aug_label)

        display(widgets.HBox([img_widget, label_widget, aug_img_widget, aug_label_widget]))

        # Save params
        self.rotate = rotate
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.scale = scale
        self.gaussian_blur = gaussian_blur
        self.gaussian_kernel_size = gaussian_kernel_size
        self.blur_sigma = blur_sigma
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_brightness_contrast = random_brightness_contrast
        self.rgb_shift = rgb_shift 

    def transform_toy(self, rotate=0, translate_x=0.1, translate_y=0.1, scale=0.1, gaussian_blur=False, gaussian_kernel_size=3, blur_sigma=0.5, horizontal_flip=False, 
                    vertical_flip=False, random_brightness_contrast=False, rgb_shift=False, file_upload=None):
        
        if file_upload:
            content = file_upload[0]['content']
            mnist_img = Image.open(io.BytesIO(content)).convert("RGB")
        else:
            # fallback
            mnist_img = Image.open("data/sample_mnist/sample_mnist_0.jpg").convert("RGB")
        buf = io.BytesIO()
        mnist_img.save(buf, format="PNG")
        widget_img = widgets.Image(value=buf.getvalue(), format='png', width=300, height=300)

        aug = AlbumentationsTransform(A.Compose([
            A.Rotate(limit=(rotate, rotate), p=1.0),
            A.Affine(translate_percent={'x': translate_x, 'y': translate_y}, scale=(scale, scale), p=1.0),
            A.GaussianBlur(blur_limit=(gaussian_kernel_size), sigma_limit=(blur_sigma, blur_sigma), p=1.0 if gaussian_blur else 0.0),
            A.HorizontalFlip(p=1.0 if horizontal_flip else 0.0),
            A.VerticalFlip(p=1.0 if vertical_flip else 0.0),
            A.RandomBrightnessContrast(p=1.0 if random_brightness_contrast else 0.0),
            A.RGBShift(p=1.0 if rgb_shift else 0.0)
        ]))
        
        aug_img = aug(mnist_img).numpy().astype(np.uint8)  
        if aug_img.ndim == 3 and aug_img.shape[0] in [1,3]:
            aug_img = np.transpose(aug_img, (1,2,0))  # CHW -> HWC

        if aug_img.shape[2] == 1:
            aug_img = aug_img[:,:,0]

        aug_img = Image.fromarray(aug_img)
        buf = io.BytesIO()
        aug_img.save(buf, format="PNG")
        aug_widget_img = widgets.Image(value=buf.getvalue(), format='png', width=250, height=250)
        
        image_box = widgets.HBox([widget_img, aug_widget_img])
        display(image_box)
        
        self.rotate = rotate
        self.translate_x=translate_x
        self.translate_y=translate_y
        self.scale=scale
        self.gaussian_blur=gaussian_blur
        self.gaussian_kernel_size=gaussian_kernel_size
        self.blur_sigma=blur_sigma
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.random_brightness_contrast=random_brightness_contrast
        self.rgb_shift=rgb_shift
        
    def retrieve_augmentation(self, p=0.5):
        return AlbumentationsTransform(A.Compose([
            A.Rotate(limit=(-1.0*self.rotate, self.rotate), p=p),
            A.Affine(translate_percent={'x': self.translate_x, 'y': self.translate_y}, scale=(self.scale, self.scale), p=p),
            A.GaussianBlur(blur_limit=(self.gaussian_kernel_size), sigma_limit=(self.blur_sigma, self.blur_sigma), p=p if self.gaussian_blur else 0.0),
            A.HorizontalFlip(p=p if self.horizontal_flip else 0.0),
            A.VerticalFlip(p=p if self.vertical_flip else 0.0),
            A.RandomBrightnessContrast(p=p if self.random_brightness_contrast else 0.0),
            A.RGBShift(p=p if self.rgb_shift else 0.0),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
        ]))
        
        
def retrieve_mnist_menu():
    viz = AugVisualizer()

    style = {'description_width': 'initial', 'handle_color': 'lightblue', 'width':'500px' }

    rotate=widgets.IntSlider(min=-180, max=180, step=1, value=0, description='Rotation Angle [-180, 180]', style=style)
    translate_x = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0, description='Translation % X-Axis', style=style)
    translate_y = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0, description='Translation % Y-Axis', style=style)
    scale = widgets.FloatSlider(min=0.5, max=1.5, step=0.1, value=1.0, style=style, description='Scale (Zoom In/Out)')
    gaussian_blur= widgets.Checkbox(description = "Apply Gaussian Blur")
    gaussian_kernel_size= widgets.FloatSlider(min=1.0, max=5.0, step=2.0, value=1.0, style=style, description='Gaussian Kernel Size')
    blur_sigma = widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=0.1, style=style, description='Gaussian Kernel Sigma')
    horizontal_flip = widgets.Checkbox(description = "Horizontal Flip")
    vertical_flip = widgets.Checkbox(description = "Vertical Flip")
    random_brightness_contrast = widgets.Checkbox(description = "Random Brightness Contrast")
    rgb_shift = widgets.Checkbox(description = "RGB Shift")
    image_upload = widgets.FileUpload(accept='.png,.jpg,.jpeg', multiple=False)
    sliders = widgets.VBox([rotate, translate_x, translate_y, scale, gaussian_blur, gaussian_kernel_size, 
                            blur_sigma, horizontal_flip, vertical_flip, random_brightness_contrast, rgb_shift])

    out = widgets.interactive_output(viz.transform_toy, 
        {
            "rotate": rotate,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "scale": scale,
            "gaussian_blur": gaussian_blur,
            "gaussian_kernel_size": gaussian_kernel_size,
            "blur_sigma": blur_sigma,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "random_brightness_contrast": random_brightness_contrast,
            "rgb_shift": rgb_shift,
            "file_upload": image_upload
        }
    )

    for slider in [rotate, translate_x, translate_y, scale, gaussian_kernel_size, blur_sigma]:
        slider.layout.width = '500px'
    menu = widgets.HBox([sliders, out, image_upload])
    menu.layout.align_items = 'center'


    return menu


        
        
def retrieve_contrails_menu():
    viz = AugVisualizer()

    style = {'description_width': 'initial', 'handle_color': 'lightblue', 'width':'500px' }

    rotate=widgets.IntSlider(min=-180, max=180, step=1, value=0, description='Rotation Angle [-180, 180]', style=style)
    translate_x = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0, description='Translation % X-Axis', style=style)
    translate_y = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=0, description='Translation % Y-Axis', style=style)
    scale = widgets.FloatSlider(min=0.5, max=1.5, step=0.1, value=1.0, style=style, description='Scale (Zoom In/Out)')
    gaussian_blur= widgets.Checkbox(description = "Apply Gaussian Blur")
    gaussian_kernel_size= widgets.FloatSlider(min=1.0, max=5.0, step=2.0, value=1.0, style=style, description='Gaussian Kernel Size')
    blur_sigma = widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=0.1, style=style, description='Gaussian Kernel Sigma')
    horizontal_flip = widgets.Checkbox(description = "Horizontal Flip")
    vertical_flip = widgets.Checkbox(description = "Vertical Flip")
    pad_mode = widgets.Dropdown(options=["None", 'reflect', 'replicate'], value="None", description='Padding Mode:', style=style)
    random_brightness_contrast = widgets.Checkbox(description = "Random Brightness Contrast")
    rgb_shift = widgets.Checkbox(description = "RGB Shift")
    sliders = widgets.VBox([rotate, translate_x, translate_y, scale, gaussian_blur, gaussian_kernel_size, 
                            blur_sigma, horizontal_flip, vertical_flip, random_brightness_contrast, rgb_shift, pad_mode])

    out = widgets.interactive_output(viz.transform_contrail, 
        {
            "rotate": rotate,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "scale": scale,
            "gaussian_blur": gaussian_blur,
            "gaussian_kernel_size": gaussian_kernel_size,
            "blur_sigma": blur_sigma,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "random_brightness_contrast": random_brightness_contrast,
            "rgb_shift": rgb_shift,
            "pad_mode": pad_mode
        }
    )
    

    for slider in [rotate, translate_x, translate_y, scale, gaussian_kernel_size, blur_sigma]:
        slider.layout.width = '400px'
    menu = widgets.HBox([sliders, out])
    menu.layout.align_items = 'center'


    return menu