cmake_minimum_required(VERSION 3.16)
project(my_exoskeleton_description)

find_package(ament_cmake REQUIRED)

# Keep the URDF and meshes canonical in the Isaac Lab checkout. The ROS package
# installs a build-time copy so Gazebo can resolve relative mesh paths and the
# controller configuration without duplicating source assets.
set(ISAACLAB_ASSET_DIR "$ENV{ISAACLAB_ASSET_DIR}" CACHE PATH
    "Path to the Isaac Lab my_robot asset directory")
set(SOURCE_URDF "${ISAACLAB_ASSET_DIR}/urdf/my_exoskeleton.urdf")
set(SOURCE_MESHES "${ISAACLAB_ASSET_DIR}/meshes")

if(NOT EXISTS "${SOURCE_URDF}")
  message(FATAL_ERROR
    "Cannot find ${SOURCE_URDF}. Set ISAACLAB_ASSET_DIR to the my_robot asset directory.")
endif()
if(NOT IS_DIRECTORY "${SOURCE_MESHES}")
  message(FATAL_ERROR "Cannot find mesh directory ${SOURCE_MESHES}.")
endif()

file(READ "${SOURCE_URDF}" URDF_CONTENT)

# The imported URDF contains the old machine-specific controller path. Replace
# it only in the ROS install copy; the Isaac Lab source URDF remains unchanged.
set(OLD_CONTROLLER_PATH
    "/home/black_paper/my_exoskeleton/install/my_exoskeleton/share/my_exoskeleton/config/controllers.yaml")
set(ROS_CONTROLLER_PATH
    "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/controllers.yaml")
string(REPLACE "${OLD_CONTROLLER_PATH}" "${ROS_CONTROLLER_PATH}"
       URDF_CONTENT "${URDF_CONTENT}")

set(GENERATED_URDF_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/urdf")
file(MAKE_DIRECTORY "${GENERATED_URDF_DIR}")
file(WRITE "${GENERATED_URDF_DIR}/my_exoskeleton.urdf" "${URDF_CONTENT}")

# Gazebo's Contact system registers sensors when a model is present while the
# world is loading. Generate an SDF model from the canonical URDF so the launch
# file can include it statically instead of spawning it after startup.
find_program(GZ_EXECUTABLE gz REQUIRED)
execute_process(
  COMMAND "${GZ_EXECUTABLE}" sdf -p "${SOURCE_URDF}"
  OUTPUT_VARIABLE GENERATED_SDF
  ERROR_VARIABLE GZ_SDF_ERRORS
  RESULT_VARIABLE GZ_SDF_RESULT
)
if(NOT GZ_SDF_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not convert ${SOURCE_URDF} to SDF: ${GZ_SDF_ERRORS}")
endif()
string(REPLACE "${OLD_CONTROLLER_PATH}" "${ROS_CONTROLLER_PATH}"
       GENERATED_SDF "${GENERATED_SDF}")
string(REPLACE "../meshes/" "../../meshes/"
       GENERATED_SDF "${GENERATED_SDF}")

set(GENERATED_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/models/my_exoskeleton")
file(MAKE_DIRECTORY "${GENERATED_MODEL_DIR}")
file(WRITE "${GENERATED_MODEL_DIR}/model.sdf" "${GENERATED_SDF}")
file(WRITE "${GENERATED_MODEL_DIR}/model.config"
  "<?xml version=\"1.0\"?>\n"
  "<model>\n"
  "  <name>my_exoskeleton</name>\n"
  "  <version>1.0</version>\n"
  "  <sdf version=\"1.9\">model.sdf</sdf>\n"
  "  <author><name>lqf</name></author>\n"
  "  <description>My exoskeleton generated from the canonical URDF.</description>\n"
  "</model>\n")

install(FILES "${GENERATED_URDF_DIR}/my_exoskeleton.urdf"
        DESTINATION share/${PROJECT_NAME}/urdf)
install(DIRECTORY "${GENERATED_MODEL_DIR}/"
        DESTINATION share/${PROJECT_NAME}/models/my_exoskeleton)
install(DIRECTORY "${SOURCE_MESHES}/"
        DESTINATION share/${PROJECT_NAME}/meshes)
install(DIRECTORY config launch worlds
        DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
        scripts/contact_grid_mapper.py
        scripts/sensor_rate_limiter.py
        DESTINATION lib/${PROJECT_NAME})

ament_package()
