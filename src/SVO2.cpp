#include <GSLAM/core/GSLAM.h>
#include <GSLAM/core/HashMap.h>
#include <GSLAM/core/Messenger.h>
#include <ceres/ceres.h>
#include <svo/common/conversions.h>
#include <svo/common/imu_calibration.h>
#include <svo/svo.h>
#include <svo/tracker/feature_tracking_utils.h>
#include <vikit/params_helper.h>
#include <opencv2/imgproc.hpp>
#include <GSLAM/core/Event.h>

using namespace svo;

BaseOptions loadBaseOptions(GSLAM::Svar pnh, bool forward_default) {
  BaseOptions o;
  o.max_n_kfs = pnh.GetInt("max_n_kfs", 5);
  o.use_imu = pnh.Get<bool>("use_imu", false);
  o.trace_dir = pnh.GetString("trace_dir", o.trace_dir);
  o.quality_min_fts = pnh.GetInt("quality_min_fts", 50);
  o.quality_max_fts_drop = pnh.GetInt("quality_max_drop_fts", 40);
  o.relocalization_max_trials = pnh.GetInt("relocalization_max_trials", 50);
  o.poseoptim_prior_lambda = pnh.GetDouble("poseoptim_prior_lambda", 0.0);
  o.poseoptim_using_unit_sphere =
      pnh.Get<bool>("poseoptim_using_unit_sphere", false);
  o.img_align_prior_lambda_rot =
      pnh.GetDouble("img_align_prior_lambda_rot", 0.0);
  o.img_align_prior_lambda_trans =
      pnh.GetDouble("img_align_prior_lambda_trans", 0.0);
  o.structure_optimization_max_pts =
      pnh.GetInt("structure_optimization_max_pts", 20);
  o.init_map_scale = pnh.GetDouble("map_scale", 1.0);
  std::string default_kf_criterion =
      forward_default ? "FORWARD" : "DOWNLOOKING";
  if (pnh.GetString("kfselect_criterion", default_kf_criterion) == "FORWARD")
    o.kfselect_criterion = KeyframeCriterion::FORWARD;
  else
    o.kfselect_criterion = KeyframeCriterion::DOWNLOOKING;
  o.kfselect_min_dist = pnh.GetDouble("kfselect_min_dist", 0.12);
  o.kfselect_numkfs_upper_thresh =
      pnh.GetInt("kfselect_numkfs_upper_thresh", 120);
  o.kfselect_numkfs_lower_thresh =
      pnh.GetDouble("kfselect_numkfs_lower_thresh", 70);
  o.kfselect_min_dist_metric = pnh.GetDouble("kfselect_min_dist_metric", 0.01);
  o.kfselect_min_angle = pnh.GetDouble("kfselect_min_angle", 20);
  o.kfselect_min_disparity = pnh.GetDouble("kfselect_min_disparity", 40);
  o.kfselect_min_num_frames_between_kfs =
      pnh.GetInt("kfselect_min_num_frames_between_kfs", 2);
  o.img_align_max_level = pnh.GetInt("img_align_max_level", 4);
  o.img_align_min_level = pnh.GetInt("img_align_min_level", 2);
  o.img_align_robustification =
      pnh.Get<bool>("img_align_robustification", false);
  o.img_align_use_distortion_jacobian =
      pnh.Get<bool>("img_align_use_distortion_jacobian", false);
  o.img_align_est_illumination_gain =
      pnh.Get<bool>("img_align_est_illumination_gain", false);
  o.img_align_est_illumination_offset =
      pnh.Get<bool>("img_align_est_illumination_offset", false);
  o.poseoptim_thresh = pnh.GetDouble("poseoptim_thresh", 2.0);
  o.update_seeds_with_old_keyframes =
      pnh.Get<bool>("update_seeds_with_old_keyframes", true);
  o.use_async_reprojectors = pnh.Get<bool>("use_async_reprojectors", false);
  return o;
}

DetectorOptions loadDetectorOptions(GSLAM::Svar pnh) {
  DetectorOptions o;
  o.cell_size = pnh.GetInt("grid_size", 35);
  o.max_level = pnh.GetInt("n_pyr_levels", 3) - 1;
  o.threshold_primary = pnh.GetInt("detector_threshold_primary", 10);
  o.threshold_secondary = pnh.GetInt("detector_threshold_secondary", 200);
  if (pnh.Get<bool>("use_edgelets", true))
    o.detector_type = DetectorType::kFastGrad;
  else
    o.detector_type = DetectorType::kFast;
  return o;
}

DepthFilterOptions loadDepthFilterOptions(GSLAM::Svar pnh) {
  DepthFilterOptions o;
  o.max_search_level = pnh.GetInt("n_pyr_levels", 3) - 1;
  o.use_threaded_depthfilter = pnh.Get<bool>("use_threaded_depthfilter", true);
  o.seed_convergence_sigma2_thresh =
      pnh.GetDouble("seed_convergence_sigma2_thresh", 200.0);
  o.scan_epi_unit_sphere = pnh.Get<bool>("scan_epi_unit_sphere", false);
  o.affine_est_offset = pnh.Get<bool>("depth_filter_affine_est_offset", true);
  o.affine_est_gain = pnh.Get<bool>("depth_filter_affine_est_gain", false);
  o.max_n_seeds_per_frame =
      pnh.GetInt("max_fts", 120) * pnh.GetDouble("max_seeds_ratio", 3.0);
  return o;
}

InitializationOptions loadInitializationOptions(GSLAM::Svar pnh) {
  InitializationOptions o;
  o.init_min_features = pnh.GetInt("init_min_features", 100);
  o.init_min_tracked = pnh.GetInt("init_min_tracked", 80);
  o.init_min_inliers = pnh.GetInt("init_min_inliers", 70);
  o.init_min_disparity = pnh.GetDouble("init_min_disparity", 40.0);
  o.init_min_features_factor = pnh.GetDouble("init_min_features_factor", 2.0);
  o.reproj_error_thresh = pnh.GetDouble("reproj_err_thresh", 2.0);
  o.init_disparity_pivot_ratio =
      pnh.GetDouble("init_disparity_pivot_ratio", 0.5);
  std::string init_method = pnh.GetString("init_method", "FivePoint");
  if (init_method == "Homography")
    o.init_type = InitializerType::kHomography;
  else if (init_method == "TwoPoint")
    o.init_type = InitializerType::kTwoPoint;
  else if (init_method == "FivePoint")
    o.init_type = InitializerType::kFivePoint;
  else if (init_method == "OneShot")
    o.init_type = InitializerType::kOneShot;
  else
    SVO_ERROR_STREAM("Initialization Method not supported: " << init_method);
  return o;
}

FeatureTrackerOptions loadTrackerOptions(GSLAM::Svar pnh) {
  FeatureTrackerOptions o;
  o.klt_max_level = pnh.GetInt("klt_max_level", 4);
  o.klt_min_level = pnh.GetInt("klt_min_level", 0.001);
  return o;
}

ReprojectorOptions loadReprojectorOptions(GSLAM::Svar pnh) {
  ReprojectorOptions o;
  o.max_n_kfs = pnh.GetInt("reprojector_max_n_kfs", 5);
  o.max_n_features_per_frame = pnh.GetInt("max_fts", 160);
  o.cell_size = pnh.GetInt("grid_size", 35);
  o.reproject_unconverged_seeds =
      pnh.Get<bool>("reproject_unconverged_seeds", true);
  o.affine_est_offset = pnh.Get<bool>("reprojector_affine_est_offset", true);
  o.affine_est_gain = pnh.Get<bool>("reprojector_affine_est_gain", false);
  return o;
}

CameraBundle::Ptr loadCameraFromYaml(GSLAM::Svar pnh) {
  std::string calib_file = pnh.GetString("calib_file", "~/cam.yaml");
  CameraBundle::Ptr ncam = CameraBundle::loadFromYaml(calib_file);
  std::cout << "loaded " << ncam->numCameras() << " cameras";
  for (const auto& cam : ncam->getCameraVector())
    cam->printParameters(std::cout, "");
  return ncam;
}

StereoTriangulationOptions loadStereoOptions(GSLAM::Svar pnh) {
  StereoTriangulationOptions o;
  o.triangulate_n_features = pnh.GetInt("max_fts", 120);
  o.max_depth_inv = pnh.GetDouble("max_depth_inv", 1.0 / 50.0);
  o.min_depth_inv = pnh.GetDouble("min_depth_inv", 1.0 / 0.5);
  o.mean_depth_inv = pnh.GetDouble("mean_depth_inv", 1.0 / 2.0);
  return o;
}

ImuHandler::Ptr getImuHandler(GSLAM::Svar pnh) {
  std::string calib_file = pnh.GetString("calib_file", "");
  ImuCalibration imu_calib = ImuHandler::loadCalibrationFromFile(calib_file);
  imu_calib.print("Loaded IMU Calibration");
  ImuInitialization imu_init =
      ImuHandler::loadInitializationFromFile(calib_file);
  imu_init.print("Loaded IMU Initialization");
  ImuHandler::Ptr imu_handler(new ImuHandler(imu_calib, imu_init));
  return imu_handler;
}

void setInitialPose(GSLAM::Svar pnh, FrameHandlerBase& vo) {
  Transformation T_world_imuinit(
      Quaternion(pnh.GetDouble("T_world_imuinit/qw", 1.0),
                 pnh.GetDouble("T_world_imuinit/qx", 0.0),
                 pnh.GetDouble("T_world_imuinit/qy", 0.0),
                 pnh.GetDouble("T_world_imuinit/qz", 0.0)),
      Vector3d(pnh.GetDouble("T_world_imuinit/tx", 0.0),
               pnh.GetDouble("T_world_imuinit/ty", 0.0),
               pnh.GetDouble("T_world_imuinit/tz", 0.0)));
  vo.setInitialImuPose(T_world_imuinit);
}

FrameHandlerMono::Ptr makeMono(GSLAM::Svar pnh,
                               const CameraBundlePtr& cam = nullptr) {
  // Create camera
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);

  // Init VO
  FrameHandlerMono::Ptr vo = std::make_shared<FrameHandlerMono>(
      loadBaseOptions(pnh, false), loadDepthFilterOptions(pnh),
      loadDetectorOptions(pnh), loadInitializationOptions(pnh),
      loadReprojectorOptions(pnh), loadTrackerOptions(pnh), ncam);

  // Get initial position and orientation of IMU
  setInitialPose(pnh, *vo);

  return vo;
}

FrameHandlerStereo::Ptr makeStereo(GSLAM::Svar pnh,
                                   const CameraBundlePtr& cam = nullptr) {
  // Load cameras
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);

  // Init VO
  InitializationOptions init_options = loadInitializationOptions(pnh);
  init_options.init_type = InitializerType::kStereo;
  FrameHandlerStereo::Ptr vo = std::make_shared<FrameHandlerStereo>(
      loadBaseOptions(pnh, true), loadDepthFilterOptions(pnh),
      loadDetectorOptions(pnh), init_options, loadStereoOptions(pnh),
      loadReprojectorOptions(pnh), loadTrackerOptions(pnh), ncam);

  // Get initial position and orientation of IMU
  setInitialPose(pnh, *vo);

  return vo;
}

FrameHandlerArray::Ptr makeArray(GSLAM::Svar pnh,
                                 const CameraBundlePtr& cam = nullptr) {
  // Load cameras
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);

  // Init VO
  InitializationOptions init_options = loadInitializationOptions(pnh);
  init_options.init_type = InitializerType::kArrayGeometric;
  init_options.init_min_disparity = 25;
  DepthFilterOptions depth_filter_options = loadDepthFilterOptions(pnh);
  depth_filter_options.verbose = true;
  FrameHandlerArray::Ptr vo = std::make_shared<FrameHandlerArray>(
      loadBaseOptions(pnh, true), depth_filter_options,
      loadDetectorOptions(pnh), init_options, loadReprojectorOptions(pnh),
      loadTrackerOptions(pnh), ncam);

  // Get initial position and orientation of IMU
  setInitialPose(pnh, *vo);

  return vo;
}

class SVO2 : public GSLAM::SLAM {
 public:
  SVO2() {
      ros::Time::init();
      setMap(GSLAM::MapPtr(new GSLAM::HashMap()));
  }

  virtual bool    setSvar(GSLAM::Svar& var){
      init(var);
  }

  GSLAM::Messenger init(GSLAM::Svar var) {
    config = var;
    google::InitGoogleLogging(var.GetString("ProgramName", "exe").c_str());
    google::InstallFailureSignalHandler();


    if (config.Get<bool>("use_imu", true)) {
        messenger.subscribe(var.GetString("imu_topic", "imu"), 10,
                            &SVO2::imuCallback, this);
    }
    messenger.subscribe(var.GetString("imgframe_topic", "images"), 2,
                        &SVO2::imagesCallback, this);

    pub_init_tracks = messenger.advertise<GSLAM::GImage>(type()+"/init_tracks", 1);
    pub_curframe = messenger.advertise<GSLAM::MapFrame>(type()+"/curframe", 1);
    pub_map = messenger.advertise<GSLAM::Map>(type()+"/map", 1);

    return messenger;
  }

  virtual std::string type() const { return "SVO2"; }
  virtual bool valid() const { return true; }
  virtual bool isDrawable() const { return false; }

  virtual bool track(GSLAM::FramePtr& frame) {
    if (frame->cameraNum() == 0)
      imuCallback(frame);
    else
      imagesCallback(frame);
    return true;
  }

  void imuCallback(const GSLAM::FramePtr& imuFrame) {
      if(!imu_handler_)
      {
          imu_handler_ = getImuHandler(config);
      }
    GSLAM::Point3d linear_acceleration, angular_velocity;
    if (!imuFrame->getAcceleration(linear_acceleration)) return;
    if (!imuFrame->getAngularVelocity(angular_velocity)) return;
    const Eigen::Vector3d omega_imu(angular_velocity.x, angular_velocity.y,
                                    angular_velocity.z);
    const Eigen::Vector3d lin_acc_imu(
        linear_acceleration.x, linear_acceleration.y, linear_acceleration.z);
    const ImuMeasurement m(imuFrame->timestamp(), omega_imu, lin_acc_imu);
    if (imu_handler_)
      imu_handler_->addImuMeasurement(m);
    else
      SVO_ERROR_STREAM("SvoNode has no ImuHandler");
  }

  void setImuPrior(const double timestamp) {
    if (imu_handler_ && !svo_->hasStarted() &&
        config.Get<bool>("set_initial_attitude_from_gravity_", true)) {
      // set initial orientation
      Quaternion R_imu_world;
      if (imu_handler_->getInitialAttitude(timestamp, R_imu_world)) {
        VLOG(3) << "Set initial orientation from accelerometer measurements.";
        svo_->setRotationPrior(R_imu_world);
      }
    } else if (imu_handler_ && svo_->getLastFrames()) {
      // set incremental rotation prior
      ImuMeasurements imu_measurements;
      if (imu_handler_->getMeasurements(
              svo_->getLastFrames()->getMinTimestampNanoseconds() *
                  common::conversions::kNanoSecondsToSeconds,
              timestamp, false, imu_measurements)) {
        Quaternion R_lastimu_newimu;
        if (imu_handler_->getRelativeRotationPrior(
                svo_->getLastFrames()->getMinTimestampNanoseconds() *
                    common::conversions::kNanoSecondsToSeconds,
                timestamp, false, R_lastimu_newimu)) {
          VLOG(3) << "Set incremental rotation prior from IMU.";
          svo_->setRotationIncrementPrior(R_lastimu_newimu);
        }
      }
    }
  }

  void imagesCallback(const GSLAM::FramePtr& imgFrame) {
      if(!svo_){
          switch (imgFrame->cameraNum()) {
          case 1:
              svo_ = makeMono(config);
              break;
          case 2:
              svo_ = makeStereo(config);
              break;
          default:
              svo_ = makeArray(config);
              break;
          }
          svo_->start();
      }
    std::vector<cv::Mat> images;
    for (int i = 0; i < imgFrame->cameraNum(); i++) {
      images.push_back(imgFrame->getImage(i));
    }

    setImuPrior(imgFrame->timestamp());

    imageCallbackPreprocessing(imgFrame->timestamp());

    svo_->addImageBundle(images, imgFrame->timestamp() * 1e9);
    publishResults(imgFrame);

    if (svo_->stage() == Stage::kPaused &&
        config.Get<bool>("automatic_reinitialization", false))
      svo_->start();

    imageCallbackPostprocessing();
  }

  void publishResults(const GSLAM::FramePtr& imgFrame) {
    CHECK_NOTNULL(svo_.get());
    switch (svo_->stage()) {
      case Stage::kTracking: {
        if (pub_curframe.getNumSubscribers() > 0||true) {
          publishCurrent(imgFrame);
        }
        if(pub_map.getNumSubscribers()>0||true){
            publishMap();
        }
        break;
      }
      case Stage::kInitializing: {
//        if (pub_init_tracks.getNumSubscribers() == 0) break;
          publishInitTracks(imgFrame);
        break;
      }
      case Stage::kPaused:
      case Stage::kRelocalization:
        break;
      default:
        LOG(FATAL) << "Unknown stage";
        break;
    }
  }

  void publishCurrent(const GSLAM::FramePtr& imgFrame){
      auto T_world_imu = svo_->getLastFrames()->get_T_W_B();
      Eigen::Quaterniond q = T_world_imu.getRotation().toImplementation();
      Eigen::Vector3d p = T_world_imu.getPosition();
      GSLAM::SE3 pose(p[0], p[1], p[2], q.x(), q.y(), q.z(), q.w());
      imgFrame->setPose(pose);
      std::cerr<<"Frame "<<imgFrame->id()<<":"<<pose;
      pub_curframe.publish(imgFrame);
      _handle->handle(new GSLAM::CurrentFrameEvent(imgFrame));
  }

  void publishMap(){
      GSLAM::MapPtr map=getMap();
      map->clear();
      MapPtr svomap=svo_->map();
      GSLAM::PointID ptid=0;
      for(auto kf : svomap->keyframes_)
      {
        const FramePtr& frame = kf.second;
        const Transformation T_w_f = frame->T_world_cam();
        for(size_t i = 0; i < frame->num_features_; ++i)
        {
          if(isSeed(frame->type_vec_[i]))
          {
            CHECK(!frame->seed_ref_vec_[i].keyframe) << "Data inconsistent";
            const Vector3d xyz = T_w_f * frame->getSeedPosInFrame(i);
            GSLAM::PointPtr pt(new GSLAM::MapPoint(++ptid,
                             GSLAM::Point3d(xyz[0],xyz[1],xyz[2])));
            map->insertMapPoint(pt);
          }
        }
      }
      _handle->handle(map);
  }

  void publishInitTracks(const GSLAM::FramePtr& imgFrame){
      const auto& frames_ref = svo_->initializer_->frames_ref_;
      const auto& frames_cur = svo_->getLastFrames();
      for (size_t i = 0; i < frames_ref->size(); ++i) {
        std::vector<std::pair<size_t, size_t>> matches_ref_cur;
        cv::Mat img = imgFrame->getImage(i).clone();
        const Keypoints& px_ref = frames_ref->at(i)->px_vec_;
        const Keypoints& px_cur = frames_cur->at(i)->px_vec_;
        for (size_t j = 0; j < matches_ref_cur.size(); ++j) {
          size_t i_ref = matches_ref_cur[i].first;
          size_t i_cur = matches_ref_cur[i].second;
          cv::line(
              img,
              cv::Point2f(px_cur(0, i_cur) , px_cur(1, i_cur) ),
              cv::Point2f(px_ref(0, i_ref) , px_ref(1, i_ref) ),
              cv::Scalar(0, 255, 0, 0), 2);
        }
        pub_init_tracks.publish(GSLAM::GImage(img));
        _handle->handle(new GSLAM::DebugImageEvent(img,"init_tracks"));
        LOG(INFO)<<"Initializing ...";
        break;
      }
  }

  // These functions are called before and after monoCallback or stereoCallback.
  // a derived class can implement some additional logic here.
  virtual void imageCallbackPreprocessing(double timestamp_nanoseconds) {}
  virtual void imageCallbackPostprocessing() {}

  GSLAM::Messenger messenger;
  GSLAM::Svar config;
  std::shared_ptr<FrameHandlerBase> svo_;
  std::shared_ptr<ImuHandler> imu_handler_;
  GSLAM::Publisher pub_init_tracks, pub_curframe, pub_map;
};

USE_GSLAM_PLUGIN(SVO2);