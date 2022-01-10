import segmentation_models_3D  as sm
from get_data_ready import ready_data
from model import get_model
from data_loader import imageLoader

def make_data_ready():
    data_path = "Change this to path of data of BRats2020"
    save_path = "Where you want to save the processed data"
    img_list,mask_list=ready_data(data_path,save_path)
    
    return save_path,img_list,mask_list

def ready_model():
    model = get_model()
    # Defining the loss,optimizer, and metrics
    wt0,wt1,wt2,wt3 = 0.25,0.25,0.25,0.25
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0,wt1,wt2,wt3]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = ['accuracy',sm.metrics.IOUScore(threshold=0.5)]
    LR = 0.0001
    optim = tf.keras.optimizers.Adam(LR)

    model.compile(loss=total_loss,
                optimizer=optim,
                metrics=metrics)
    return model

path_to_img,img_list,mask_list = make_data_ready()

dataset = imageLoader(path_to_img,img_list,path_to_img,mask_list,batch_size=2)

model = ready_model()

history = model.fit(dataset,steps_per_epochs=100, epochs=100)

model.save(path_to_img+'brain_tumor_segmentation.h5')