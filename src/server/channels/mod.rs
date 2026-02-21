//! Cloud Channels Module
//!
//! Provides interfaces for cloud TTS providers:
//! - OpenAI
//! - Google Cloud
//! - Aliyun (Alibaba Cloud)
//! - Volcengine (火山引擎)
//! - Minimax

pub mod traits;
pub mod registry;
pub mod error;
pub mod local;
pub mod openai;
pub mod google;
pub mod aliyun;
pub mod volcengine;
// pub mod minimax;

pub use traits::*;
pub use registry::ChannelRegistry;
pub use local::LocalTtsEngine;
pub use openai::OpenAiChannel;
pub use google::GoogleCloudChannel;
pub use aliyun::AliyunChannel;
pub use volcengine::VolcengineChannel;
