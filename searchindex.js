Search.setIndex({docnames:["Notes/tutorials/Interpretability","Notes/tutorials/Segmentation","Notes/tutorials/Transfer Learning","Notes/tutorials/installation","README","benchmark","glasses","glasses.data","glasses.data.visualisation","glasses.interpretability","glasses.models","glasses.models.base","glasses.models.classification","glasses.models.classification.alexnet","glasses.models.classification.base","glasses.models.classification.deit","glasses.models.classification.densenet","glasses.models.classification.efficientnet","glasses.models.classification.fishnet","glasses.models.classification.mobilenet","glasses.models.classification.regnet","glasses.models.classification.resnest","glasses.models.classification.resnet","glasses.models.classification.resnetxt","glasses.models.classification.senet","glasses.models.classification.vgg","glasses.models.classification.vit","glasses.models.classification.wide_resnet","glasses.models.segmentation","glasses.models.segmentation.base","glasses.models.segmentation.fpn","glasses.models.segmentation.unet","glasses.models.utils","glasses.nn","glasses.nn.activation","glasses.nn.att","glasses.nn.blocks","glasses.nn.pool","glasses.nn.regularization","glasses.utils","glasses.utils.weights","index","models_table","modules","setup","transfer_weights"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["Notes/tutorials/Interpretability.md","Notes/tutorials/Segmentation.md","Notes/tutorials/Transfer Learning.md","Notes/tutorials/installation.md","README.rst","benchmark.rst","glasses.rst","glasses.data.rst","glasses.data.visualisation.rst","glasses.interpretability.rst","glasses.models.rst","glasses.models.base.rst","glasses.models.classification.rst","glasses.models.classification.alexnet.rst","glasses.models.classification.base.rst","glasses.models.classification.deit.rst","glasses.models.classification.densenet.rst","glasses.models.classification.efficientnet.rst","glasses.models.classification.fishnet.rst","glasses.models.classification.mobilenet.rst","glasses.models.classification.regnet.rst","glasses.models.classification.resnest.rst","glasses.models.classification.resnet.rst","glasses.models.classification.resnetxt.rst","glasses.models.classification.senet.rst","glasses.models.classification.vgg.rst","glasses.models.classification.vit.rst","glasses.models.classification.wide_resnet.rst","glasses.models.segmentation.rst","glasses.models.segmentation.base.rst","glasses.models.segmentation.fpn.rst","glasses.models.segmentation.unet.rst","glasses.models.utils.rst","glasses.nn.rst","glasses.nn.activation.rst","glasses.nn.att.rst","glasses.nn.blocks.rst","glasses.nn.pool.rst","glasses.nn.regularization.rst","glasses.utils.rst","glasses.utils.weights.rst","index.rst","models_table.rst","modules.rst","setup.rst","transfer_weights.rst"],objects:{"":{glasses:[6,0,0,"-"]},"glasses.data":{visualisation:[8,0,0,"-"]},"glasses.interpretability":{GradCam:[9,0,0,"-"],Interpretability:[9,0,0,"-"],SaliencyMap:[9,0,0,"-"],ScoreCam:[9,0,0,"-"],utils:[9,0,0,"-"]},"glasses.interpretability.GradCam":{GradCam:[9,1,1,""],GradCamResult:[9,1,1,""],__call__:[9,2,1,""]},"glasses.interpretability.GradCam.GradCam":{__call__:[9,2,1,""]},"glasses.interpretability.GradCam.GradCamResult":{show:[9,2,1,""]},"glasses.interpretability.Interpretability":{Interpretability:[9,1,1,""]},"glasses.interpretability.SaliencyMap":{SaliencyMap:[9,1,1,""],SaliencyMapResult:[9,1,1,""],__call__:[9,2,1,""],guide:[9,2,1,""]},"glasses.interpretability.SaliencyMap.SaliencyMap":{__call__:[9,2,1,""],guide:[9,2,1,""]},"glasses.interpretability.SaliencyMap.SaliencyMapResult":{show:[9,2,1,""]},"glasses.interpretability.ScoreCam":{ScoreCam:[9,1,1,""],__call__:[9,2,1,""]},"glasses.interpretability.ScoreCam.ScoreCam":{__call__:[9,2,1,""]},"glasses.interpretability.utils":{convert_to_grayscale:[9,3,1,""],find_first_layer:[9,3,1,""],find_last_layer:[9,3,1,""],image2cam:[9,3,1,""],tensor2cam:[9,3,1,""]},"glasses.models":{AutoTransform:[10,0,0,"-"],base:[11,0,0,"-"]},"glasses.models.AutoTransform":{AutoTransform:[10,1,1,""],Transform:[10,1,1,""]},"glasses.models.AutoTransform.AutoTransform":{from_name:[10,2,1,""],names:[10,2,1,""],zoo:[10,4,1,""]},"glasses.models.AutoTransform.Transform":{interpolations:[10,4,1,""]},"glasses.models.base":{Encoder:[11,1,1,""],VisionModule:[11,1,1,""],protocols:[11,0,0,"-"]},"glasses.models.base.Encoder":{features:[11,2,1,""],features_widths:[11,2,1,""],stages:[11,2,1,""],training:[11,4,1,""]},"glasses.models.base.VisionModule":{summary:[11,2,1,""],training:[11,4,1,""]},"glasses.models.base.protocols":{Freezable:[11,1,1,""],Interpretable:[11,1,1,""]},"glasses.models.base.protocols.Freezable":{freeze:[11,2,1,""],set_requires_grad:[11,2,1,""],unfreeze:[11,2,1,""]},"glasses.models.base.protocols.Interpretable":{interpret:[11,2,1,""]},"glasses.models.classification":{base:[14,0,0,"-"]},"glasses.models.classification.base":{ClassificationModule:[14,1,1,""]},"glasses.models.classification.base.ClassificationModule":{forward:[14,2,1,""],initialize:[14,2,1,""],training:[14,4,1,""]},"glasses.nn":{ChannelSE:[33,1,1,""],Conv2dPad:[33,1,1,""],ConvBnAct:[33,1,1,""],DropBlock:[33,1,1,""],EfficientChannelAtt:[33,1,1,""],SpatialChannelSE:[33,1,1,""],SpatialPyramidPool:[33,1,1,""],SpatialSE:[33,1,1,""],StochasticDepth:[33,1,1,""],activation:[34,0,0,"-"],att:[35,0,0,"-"],blocks:[36,0,0,"-"],pool:[37,0,0,"-"],regularization:[38,0,0,"-"]},"glasses.nn.ChannelSE":{forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.Conv2dPad":{bias:[33,4,1,""],dilation:[33,4,1,""],forward:[33,2,1,""],groups:[33,4,1,""],kernel_size:[33,4,1,""],out_channels:[33,4,1,""],output_padding:[33,4,1,""],padding:[33,4,1,""],padding_mode:[33,4,1,""],stride:[33,4,1,""],transposed:[33,4,1,""],weight:[33,4,1,""]},"glasses.nn.ConvBnAct":{training:[33,4,1,""]},"glasses.nn.DropBlock":{calculate_gamma:[33,2,1,""],forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.EfficientChannelAtt":{forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.SpatialChannelSE":{forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.SpatialPyramidPool":{forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.SpatialSE":{forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.StochasticDepth":{forward:[33,2,1,""],training:[33,4,1,""]},"glasses.nn.att":{ChannelSE:[35,1,1,""],EfficientChannelAtt:[35,1,1,""],SpatialChannelSE:[35,1,1,""],SpatialSE:[35,1,1,""],WithAtt:[35,1,1,""]},"glasses.nn.att.ChannelSE":{forward:[35,2,1,""],training:[35,4,1,""]},"glasses.nn.att.EfficientChannelAtt":{forward:[35,2,1,""],training:[35,4,1,""]},"glasses.nn.att.SpatialChannelSE":{forward:[35,2,1,""],training:[35,4,1,""]},"glasses.nn.att.SpatialSE":{forward:[35,2,1,""],training:[35,4,1,""]},"glasses.nn.blocks":{BnActConv:[36,1,1,""],Conv2dPad:[36,1,1,""],ConvBnAct:[36,1,1,""],ConvBnDropAct:[36,1,1,""],Lambda:[36,1,1,""],residuals:[36,0,0,"-"]},"glasses.nn.blocks.BnActConv":{training:[36,4,1,""]},"glasses.nn.blocks.Conv2dPad":{bias:[36,4,1,""],dilation:[36,4,1,""],forward:[36,2,1,""],groups:[36,4,1,""],kernel_size:[36,4,1,""],out_channels:[36,4,1,""],output_padding:[36,4,1,""],padding:[36,4,1,""],padding_mode:[36,4,1,""],stride:[36,4,1,""],training:[36,4,1,""],transposed:[36,4,1,""],weight:[36,4,1,""]},"glasses.nn.blocks.ConvBnAct":{training:[36,4,1,""]},"glasses.nn.blocks.ConvBnDropAct":{training:[36,4,1,""]},"glasses.nn.blocks.Lambda":{forward:[36,2,1,""],training:[36,4,1,""]},"glasses.nn.blocks.residuals":{Cat2d:[36,5,1,""],InputForward:[36,1,1,""],Residual:[36,1,1,""],ResidualAdd:[36,1,1,""],add:[36,3,1,""]},"glasses.nn.blocks.residuals.InputForward":{forward:[36,2,1,""],training:[36,4,1,""]},"glasses.nn.blocks.residuals.Residual":{forward:[36,2,1,""],training:[36,4,1,""]},"glasses.nn.blocks.residuals.ResidualAdd":{training:[36,4,1,""]},"glasses.nn.pool":{SpatialPyramidPool:[37,0,0,"-"]},"glasses.nn.pool.SpatialPyramidPool":{SpatialPyramidPool:[37,1,1,""],forward:[37,2,1,""],training:[37,4,1,""]},"glasses.nn.pool.SpatialPyramidPool.SpatialPyramidPool":{forward:[37,2,1,""],training:[37,4,1,""]},"glasses.nn.regularization":{DropBlock:[38,1,1,""],StochasticDepth:[38,1,1,""]},"glasses.nn.regularization.DropBlock":{calculate_gamma:[38,2,1,""],forward:[38,2,1,""],training:[38,4,1,""]},"glasses.nn.regularization.StochasticDepth":{forward:[38,2,1,""],training:[38,4,1,""]},"glasses.utils":{ModuleTransfer:[39,0,0,"-"],Storage:[39,0,0,"-"],Tracker:[39,0,0,"-"],weights:[40,0,0,"-"]},"glasses.utils.ModuleTransfer":{ModuleTransfer:[39,1,1,""]},"glasses.utils.ModuleTransfer.ModuleTransfer":{__call__:[39,2,1,""],dest:[39,4,1,""],dest_skip:[39,4,1,""],src:[39,4,1,""],src_skip:[39,4,1,""],verbose:[39,4,1,""]},"glasses.utils.Storage":{BackwardModuleStorage:[39,1,1,""],ForwardModuleStorage:[39,1,1,""],ModuleStorage:[39,1,1,""],MutipleKeysDict:[39,1,1,""]},"glasses.utils.Storage.ModuleStorage":{clear:[39,2,1,""],hook:[39,2,1,""],keys:[39,2,1,""],layers:[39,2,1,""],names:[39,2,1,""],register_hooks:[39,2,1,""]},"glasses.utils.Tracker":{Tracker:[39,1,1,""]},"glasses.utils.Tracker.Tracker":{handles:[39,4,1,""],module:[39,4,1,""],parametrized:[39,2,1,""],traced:[39,4,1,""]},"glasses.utils.weights":{HFModelHub:[40,0,0,"-"],PretrainedWeightsProvider:[40,0,0,"-"]},"glasses.utils.weights.HFModelHub":{HFModelHub:[40,1,1,""]},"glasses.utils.weights.HFModelHub.HFModelHub":{from_pretrained:[40,2,1,""],push_to_hub:[40,2,1,""],save_pretrained:[40,2,1,""]},"glasses.utils.weights.PretrainedWeightsProvider":{PretrainedWeightsProvider:[40,1,1,""],load_pretrained_model:[40,3,1,""],pretrained:[40,3,1,""]},"glasses.utils.weights.PretrainedWeightsProvider.PretrainedWeightsProvider":{BASE_DIR:[40,4,1,""],override:[40,4,1,""],save_dir:[40,4,1,""],verbose:[40,4,1,""]},glasses:{data:[7,0,0,"-"],interpretability:[9,0,0,"-"],nn:[33,0,0,"-"],utils:[39,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:data"},terms:{"0x7fa343f358b0":0,"0x7fd12a231eb0":11,"1000":14,"128":[1,2,36],"192":1,"224":[2,9,10,11],"2240":[2,10],"2250":[2,10],"2290":[2,10],"240":10,"256":[1,2,10,33,37],"260":10,"280":10,"300":10,"3128":40,"320":10,"380":10,"384":[1,10],"4012":40,"4060":[2,10],"416":10,"456":10,"4560":[2,10],"461":10,"4850":[2,10],"512":[1,2],"528":10,"5df126b679d7570ad2044f3":0,"600":10,"672":10,"700":0,"800":10,"case":[0,1,40],"class":[0,9,10,11,14,33,35,36,37,38,39,40],"default":[9,10,11,33,35,36,37,38,40],"final":2,"float":[10,33,36,38],"function":[2,9,14,33,35,36,37,38,40],"import":[0,1,2,33,35,39],"int":[9,10,14,33,35,36,37,38,39,40],"new":[2,40],"return":[1,2,9,10,11,33,38,39,40],"static":[10,11,40],"super":[11,33,35],"true":[9,33,35,36,39,40],"try":40,"while":[14,33,35,36,37,38,40],For:1,The:[1,9,10,33,35,36,37,38,39,40],Then:0,There:[0,39],Useful:11,Will:40,__call__:[9,39],__init__:[11,33,35],access:[0,10,11],account:40,act:[33,36],action:0,activ:[33,35,36],adapt:2,adaptivemaxpool2d:[33,37],add:[0,33,35,36,40],add_modul:[33,35],add_two:36,addit:[33,36],advanc:[33,35],affin:[33,36],after:[9,35],afterward:[14,33,35,36,37,38],aggr_func:36,aggreg:36,alexnet:12,all:[0,1,9,11,14,33,35,36,37,38,39,40],allow:[0,11,36,39,40],alreadi:[2,40],also:[0,1,2,33,35,36],although:[14,33,35,36,37,38],alwai:1,ani:[1,40],anoth:39,anystr:40,api:1,append:40,appli:[0,33,35,36],arg:[1,9,11,33,35,36,39],argmax:9,argument:40,arr:9,arrai:39,artifact:40,assert:2,assum:39,att:33,attempt:40,attent:[33,35],attributeerror:40,author:[33,35,40],auto:[0,33,36],automat:1,automodel:[0,1,2],autoreload:[4,41],autotransform:[0,2],avail:[0,2],awesom:40,axesimag:0,back:40,backwardmodulestorag:39,bar:40,base:[9,10,12,28,33,35,36,37,38,39,40],base_dir:40,batch:0,batchnorm2d:[11,33,36],batchnorm:[33,36],bearer:40,becaus:2,becom:36,befor:36,benchmark:43,bert:40,beta:[33,35],bia:[33,36,39],bicub:10,bigger:[33,35,38],bilinear:[2,10],block:[33,35,38,39],block_siz:[33,38],bnactconv:36,bool:[9,11,14,33,35,36,37,38,40],booth:[33,35,39],both:[11,14,33,35,36,37,38,40],branch:40,built:[33,36],bytesio:0,cach:40,cache_dir:40,calcul:[33,35,36],calculate_gamma:[33,38],call:[2,9,14,33,35,36,37,38],callabl:[9,10,36,40],cam:[0,9],can:[0,1,2,3,9,10,33,35,36,40],care:[14,33,35,36,37,38],cat2d:36,cat:36,centercrop:[2,10],cfg:[0,2],chang:[2,40],channel:[1,33,35],channels:[33,35],cla:11,classif:[1,9,10,33,35],classificationmodul:14,clear:39,cli:40,close:[33,38],cluster:[33,38],cnn:9,code:1,collect:39,com:[0,1,2,3,9],combin:[33,35],commit:40,commit_messag:40,compact:[4,41],complet:[33,38],compos:[1,2,10,36],comput:[4,14,33,35,36,37,38,41],concaten:36,concis:[4,41],concurr:[33,35],config:[0,2,10,40],configur:[2,40],connect:36,contain:[0,2,33,36,40],content:0,conv2d:[9,11,33,35,36],conv2dpad:[33,36],conv3x3bnact:[33,36],conv:[2,9,33,36],convact:[33,36],convbn:[33,36],convbnact:[33,36],convbndropact:36,convert:9,convert_to_grayscal:9,convolut:[2,9,33,35,36,37,38],cool:2,correct:[1,2],correctli:0,cpu:40,creat:[0,1,33,35],credit:9,crucial:2,cse_resnet50:10,cseresnet50:[33,35],ctx:9,custom:[1,36],customiz:[4,41],cute:0,cv2im:9,data:[6,41],dataset:2,dbmdz:40,deactiv:40,debug:39,decod:1,deep:[2,4,9,33,35,37,38,41],def:[1,11,33,35,39],defin:[14,33,35,36,37,38,39],deit:12,deit_base_patch16_224:10,deit_base_patch16_384:10,deit_small_patch16_224:10,deit_tiny_patch16_224:10,delet:40,densenet:12,depth:[33,38],deriv:9,descript:[9,10,11,33,35,36,40],dest:39,dest_skip:39,devic:11,dict:40,dictionari:40,differ:[0,1,2,33,36,39],dilat:[33,36],dim:0,dimens:[33,37,38],dir:40,direcli:[33,35],directli:[1,11],directori:40,doe:36,dog:0,done:[0,1,2],download:40,downsampl:[33,35],drop:[33,38],dropblock:[33,36,38],dropout:[33,38,40],due:1,dure:40,dynam:[33,36],each:[0,1,2,40],easi:1,easili:[1,2],eca:[33,35],eca_resnet50:[33,35],ecaresnet50:[33,35],effect:[33,38],effici:[33,35],efficientchannelatt:[33,35],efficientnet:12,efficientnet_b0:10,efficientnet_b1:[1,10],efficientnet_b2:10,efficientnet_b3:10,efficientnet_b4:10,efficientnet_b5:10,efficientnet_b6:10,efficientnet_b7:10,efficientnet_b8:10,efficientnet_l2:10,efficientnet_lite0:10,efficientnet_lite1:10,efficientnet_lite2:10,efficientnet_lite3:10,efficientnet_lite4:10,either:40,encod:[2,9,11,14],end:40,endpoint:40,eps:[33,36],eval:40,evalu:40,even:[2,40],everi:[14,33,35,36,37,38],exampl:[1,9,10,11,33,35,36,37,39,40],excit:[33,35],exclud:40,exist:40,expect:1,explan:9,eyes:39,factori:39,fals:[11,33,36,39,40],far:0,featur:[1,2,11,33,35,36,37],features_width:11,field:2,fig:0,figur:[0,9],file:40,find_first_lay:9,find_last_lay:9,fine:0,first:[0,9,40],fishnet:12,fix:[33,37],flat:39,fly:1,follow:[33,35,38],foo:40,forc:40,force_download:40,format:0,former:[14,33,35,36,37,38],forward:[14,33,35,36,37,38,39],forwardmodulestorag:39,found:[1,10],four:1,fpn:28,francescosaveriozuppichini:[0,1,2,3],freez:11,freezabl:11,from:[0,1,2,9,10,11,33,35,39,40],from_encod:1,from_nam:[0,1,2,10],from_pretrain:[0,1,2,40],fulli:[33,35],functool:[1,10,33,35,36],further:[33,35],gamma:[33,35,38],gener:[33,37,40],german:40,get:[0,1,2,39,40],get_encod:1,git:[0,1,2,3,40],github:[0,1,2,3,9],given:[9,10,39],glass:[0,1,2,3],going:2,grad:[0,2,9],gradcam:0,gradcamresult:9,gradient:9,grayscal:9,grayscale_im:9,group:[33,36],guid:9,half:1,handl:39,has:[1,2,39],have:[0,1,2,39,40],head:[1,14,40],higher:[33,35],home:40,hood:39,hook:[14,33,35,36,37,38,39],host:40,hostnam:40,hot:9,how:[1,2,39],http:[0,1,2,3,9,40],hub:40,huge:2,huggingfac:40,huggingface_hub:40,idea:[33,35,38],ident:36,identifi:40,ids:40,ignor:[14,33,35,36,37,38],imag:[0,2,9,33,36,37,38],image2cam:9,imagenet:[0,1,2],imagenettransformur:10,img:9,implement:[0,9,33,35,37,38],improv:39,imshow:0,in_channel:[1,14],in_featur:[2,33,35,36,39],incas:40,incomplet:40,increas:[1,33,35],index:41,info:[0,1],inherit:1,initi:[11,14,33,35,36,37,38,40],inner:11,inplac:[33,35,36],input:[1,2,9,10,11,33,35,36,38,39],input_s:10,input_shap:11,inputforward:36,insid:[0,9,33,35,40],instal:[0,1,2,41],instanc:[0,1,11,14,33,35,36,37,38],instanti:40,instead:[14,33,35,36,37,38],intern:[11,14,33,35,36,37,38],interpol:[2,10],interpolationmod:10,interpret:[6,11,41],invert:0,its:[33,36],jpeg:0,just:[0,2],keep:1,kei:39,kernel_s:[11,33,35,36],keyword:40,know:2,known:36,kwag:36,kwarg:[1,9,11,14,33,35,36,39,40],lambd:36,lambda:[36,40],last:9,latter:[14,33,35,36,37,38],layer:[1,9,11,33,36,37,38,39],learn:[4,33,35,41],learnabl:39,length:[33,37],let:0,level:40,librari:[4,41],like:[0,1,40],linear:[2,39],list:[2,10,33,37,39],load:[0,1,40],load_ext:[4,41],load_pretrained_model:40,local:[9,40],local_files_onli:40,locat:40,login:40,look:40,loop:39,lot:2,mai:2,main:[33,38,40],mani:2,map:[0,9],map_loc:40,mask:[33,38],match:1,matplotlib:[0,9],mean:[0,2,10],messag:40,method:[0,1,11,33,38],mobilenet:12,mode:[33,36,38,40],model:[0,1,6,9,33,35,39,40,41],model_a:39,model_b:39,model_id:40,model_kwarg:40,models_t:43,modifi:[33,35],modul:[1,41,43],module_b:39,modulelist:36,modulestorag:39,momentum:[33,36],more:[1,33,35],multipl:[36,39],must:40,mutiplekeysdict:39,my_model_directori:40,mymodel:11,n_class:[1,2,14],name:[2,10,39,40],namespac:40,need:[0,1,2,14,33,35,36,37,38],net:[1,33,35],network:[1,2,9,33,35,37,38],neural:[2,9,33,35],none:[9,11,33,35,36,40],normal:[0,2,9,10,33,36,38],noth:36,notic:1,now:[2,39],num_pool:[33,37],number:[2,33,35,36,37],numpi:9,numpy_arr:9,object:[2,9,10,11,35,39,40],obtain:2,of_typ:9,one:[1,9,10,11,14,33,35,36,37,38,39],ones:39,onli:[1,11,39,40],open:0,oper:39,option:[9,11,33,35,36,37,38,40],ordereddict:39,organ:40,origin:[2,33,35],other:[1,40],our:2,out:[1,39],out_channel:[33,36],out_featur:[33,35,36,39],output:[0,1,2,9,33,36,37,39,40],output_pad:[33,36],overrid:40,overridden:[14,33,35,36,37,38],own:[33,35],packag:43,pad:[33,36],padding_mod:[33,36],paid:40,panopt:1,paper:[33,38],param:[2,33,36,39],paramet:[1,2,9,10,11,33,35,36,37,38,40],parametr:39,partial:[1,10,33,35,36],pass:[0,1,2,14,33,35,36,37,38,39,40],path:40,pathlib:40,pathlik:40,per:39,perform:[2,14,33,35,36,37,38,39],pfpn:1,pictur:[0,33,38],pil:[0,2],pip:[0,1,2,3],pixel:[33,38],plt:0,pool:33,posixpath:40,post:[0,9],postpreocess:9,postprocess:[0,9],pre:[2,40],preactiv:36,preprocess:[0,2,10],pretrain:[0,40],pretrained_model_name_or_path:40,pretrained_state_dict:40,print:39,privat:40,probabl:[33,38],procedur:2,process:[0,9],properli:2,properti:[11,39],propos:[9,33,35,38],protocol:[0,40],provid:[33,36,40],proxi:40,push:40,push_to_hub:40,pyplot:[0,9],pyramid:[1,33,37],python:39,pytorch:[2,9,36,40],rais:40,rand:[9,36],randn:[1,33,37],ratio:[33,35],readi:2,receiv:40,recip:[14,33,35,36,37,38],recognit:[33,37],reduc:[1,33,35],reduced_featur:[33,35],reduct:[33,35],reg:36,regardless:[33,37],regist:[14,33,35,36,37,38,39],register_hook:39,regnet:12,regular:[33,36],relu:[11,33,35,36,39],remot:40,replac:1,repo:40,repo_url:40,report:[33,35],represent:[33,37],request:[0,40],requir:40,requires_grad:2,res:36,res_func:36,residualadd:36,resiz:[2,10],resnest200:10,resnest269:10,resnest:12,resnet18:[0,9,10,11,40],resnet26:10,resnet26d:10,resnet34:2,resnet50:[1,33,35],resnet50d:10,resnet:[1,2,9,11,12,33,35],resnetbasicblock:[33,35],resnetbottleneckblock:[33,35],resnetencod:1,resnethead:2,resnetstem:1,resnetxt:12,resolut:1,resp:40,rest:1,result:[9,36],resum:40,resume_download:40,retriev:40,revis:40,rgb:9,root:[0,1,40],run:[9,11,14,33,35,36,37,38,40],rune:3,runner:40,salienc:[0,9],saliency_map:9,saliencymap:0,saliencymapresult:9,same:[1,2,33,36,39,40],save:40,save_dir:40,save_directori:40,save_pretrain:40,scenarion:[33,35],score:9,scriptmodul:[11,14,33,35,36,37,38],scseresnet50:[33,35],se_resnet50:[33,35],see:[0,2,9,40],segment:[10,41],segmentationmodul:1,self:[11,33,35,39],senet:12,senetbasicblock:[33,35],sequenti:[11,33,35,36,39],seresnet50:[33,35],server:40,set:[39,40],set_requires_grad:11,setup:43,shape:[1,9,33,36,37],share:[1,11,14,33,35,36,37,38],shortcut:36,should:[10,11,14,33,35,36,37,38,40],show:[0,2,9,33,35,38],silent:[14,33,35,36,37,38],similar:[1,33,38],simpl:[0,33,35],simpli:40,sinc:[14,33,35,36,37,38,40],size:[1,2,10,33,36,37],skip:[33,38],small:1,softmax:9,some:40,someth:36,sourc:[9,10,11,14,33,35,36,37,38,39,40],spatial:[33,35,37],spatialchannels:[33,35],spatialpyramidpool:33,spatials:[33,35],specif:[1,2,11,40],specifi:[33,35,40],speed:2,squeez:[33,35],src:39,src_skip:39,stack:[33,36],stage:[1,11],standard:[1,40],start:[0,1,2],start_featur:1,state:[11,14,33,35,36,37,38,40],state_dict:40,std:[0,2,10],stem:1,stochast:[33,38],stochasticdepth:[33,38],store:40,str:[10,33,36,40],strict:40,stride:[33,36],string:40,sub:[1,40],subclass:[11,14,33,35,36,37,38],subset:1,suggest:2,summari:[2,11,33,35],sure:2,system:40,tada:0,tag:40,take:[1,2,14,33,35,36,37,38],target:9,techniqu:[0,9],tell:[2,40],tensor2cam:9,tensor:[2,9,10,11,14,33,35,36,37,38,39,40],tensorflow:[33,36],than:[1,2],thei:[0,39,40],them:[0,1,14,33,35,36,37,38],thi:[0,1,2,14,33,35,36,37,38,39,40],thing:39,three:1,through:[0,1,2],thu:1,time:2,titl:0,token:40,top:[33,36],torch:[1,9,11,14,33,35,36,37,38,39,40],torchinfo:11,torchvis:[0,10],totensor:[2,10],trace:39,track:39,track_running_stat:[33,36],train:[2,11,14,33,35,36,37,38,40],tran:39,transfer:[39,41],transfer_weight:43,transform:[0,2,10,40],transpos:[33,36],tupl:[10,11,33,36],tutori:2,twice:1,two:39,type:[9,10,11,33,36,38,39,40],uncas:40,under:[39,40],understand:0,unet:[1,10,28],unetdecod:1,unfreez:11,union:40,unsqueez:0,upsampl:1,url:40,usag:[35,40],use:[0,1,2,9,10,33,35,36,40],use_auth_token:40,used:[1,9,33,35,40],useful:2,user:40,uses:[2,11],using:[0,1,2,3,11,33,35,39,40],usual:[2,36],util:[6,10,33,35,36,41],utkuozbulak:9,valid:40,valu:39,verbos:[39,40],veri:[33,35],version:40,vgg:12,via:9,vision:[4,11,41],visionmodul:[0,11,14],visual:[9,33,37],visualis:[7,9,33,35],vit:12,vit_base_patch16_224:10,vit_base_patch16_384:10,vit_base_patch32_384:10,vit_huge_patch16_224:10,vit_huge_patch32_384:10,vit_large_patch16_224:10,vit_large_patch16_384:10,vit_large_patch32_384:10,vit_small_patch16_224:10,wai:[1,2,39],want:[0,40],webp:0,weight:[0,1,2,9,11,33,36,39],were:39,what:1,when:[2,40],where2lay:39,where:[2,33,35,36],whether:40,which:40,who:[2,11],why:1,wide_resnet:12,width:[0,1,2],wil:9,wise:[33,35],wish:9,withatt:[33,35],within:[14,33,35,36,37,38],word:1,work:1,would:[0,1],wrong:40,you:[0,1,2,3,9,10,33,35,36,40],your:[2,10,33,35,40],your_model:40,zoo:10},titles:["Interpretability","Segmentation","Transfer Learning","Installation","Glasses \ud83d\ude0e","benchmark module","glasses package","glasses.data package","glasses.data.visualisation package","glasses.interpretability package","glasses.models package","glasses.models.base package","glasses.models.classification package","glasses.models.classification.alexnet package","glasses.models.classification.base package","glasses.models.classification.deit package","glasses.models.classification.densenet package","glasses.models.classification.efficientnet package","glasses.models.classification.fishnet package","glasses.models.classification.mobilenet package","glasses.models.classification.regnet package","glasses.models.classification.resnest package","glasses.models.classification.resnet package","glasses.models.classification.resnetxt package","glasses.models.classification.senet package","glasses.models.classification.vgg package","glasses.models.classification.vit package","glasses.models.classification.wide_resnet package","glasses.models.segmentation package","glasses.models.segmentation.base package","glasses.models.segmentation.fpn package","glasses.models.segmentation.unet package","glasses.models.utils package","glasses.nn package","glasses.nn.activation package","glasses.nn.att package","glasses.nn.blocks package","glasses.nn.pool package","glasses.nn.regularization package","glasses.utils package","glasses.utils.weights package","Glasses \ud83d\ude0e","models_table module","glasses","setup module","transfer_weights module"],titleterms:{activ:34,alexnet:13,att:35,automodel:10,autotransform:10,base:[11,14,29],benchmark:5,block:36,chang:1,classif:[2,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],content:[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],data:[7,8],deit:15,densenet:16,efficientnet:17,encod:1,fishnet:18,fpn:30,freez:2,glass:[4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43],gradcam:9,head:2,hfmodelhub:40,indic:41,instal:3,interpret:[0,9],layer:2,learn:2,load:2,mobilenet:19,model:[2,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],models_t:42,modul:[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,45],moduletransf:39,note:41,packag:[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],pool:37,preambula:[0,1,2],pretrain:1,pretrainedweightsprovid:40,protocol:11,refer:41,regnet:20,regular:38,replac:2,residu:36,resnest:21,resnet:22,resnetxt:23,saliencymap:9,scaler:32,scorecam:9,segment:[1,28,29,30,31],senet:24,setup:44,spatialpyramidpool:37,storag:39,submodul:[9,10,11,32,36,37,39,40],subpackag:[6,7,10,12,28,33,39],tabl:41,tracker:39,transfer:2,transfer_weight:45,unet:31,util:[9,32,39,40],vgg:25,visualis:8,vit:26,weight:40,wide_resnet:27}})