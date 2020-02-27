import spectral.io.envi as envi
import spectral

spectral.settings.WX_GL_DEPTH_SIZE = 16

img = envi.open('/home/juanval/Hyper/Experiment_2/Tommy/control_group/sample_1/tom_cgs_01_16000_us_2x_2019-11-24T121227_corr.hdr', '/home/juanval/Hyper/Experiment_2/Tommy/control_group/sample_1/tom_cgs_01_16000_us_2x_2019-11-24T121227_corr.hyspex')


img.info()


#view = imshow(img)
view_nd(img)
#save_rgb('rgb.jpg', img, [29, 19, 9])
