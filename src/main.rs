use adw::prelude::*;
use anyhow::Error;
use byte_slice_cast::*;
use clap::Parser;
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_audio as gst_audio;
use gtk::{gdk, gio, glib, pango};
use gtk4 as gtk;
use log::*;
use std::collections::VecDeque;
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to model file
    #[arg(short, long, default_value = "models/ggml-base.en.bin")]
    model: String,

    /// spoken language
    #[arg(short, long)]
    language: Option<String>,

    /// audio length (ms)
    #[arg(short, long, default_value_t = 10000)]
    length: usize,

    /// audio step size (ms)
    #[arg(short, long, default_value_t = 2000)]
    step: u64,

    /// audio to keep from previous step (ms)
    #[arg(short, long, default_value_t = 200)]
    keep: usize,

    /// window height
    #[arg(long, default_value_t = 100)]
    height: i32,

    /// window width
    #[arg(short, long, default_value_t = 1024)]
    width: i32,

    /// font size
    #[arg(short, long, default_value_t = 24)]
    font_size: i32,

    /// gstreamer source name to listen on
    #[arg(long, default_value = "pipewiresrc")]
    source: String,

    /// log verbosity (-v, -vv...)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn create_pipeline(
    source: &str,
    buf: Arc<Mutex<VecDeque<f32>>>,
    buf_size: usize,
) -> Result<gst::Pipeline, Error> {
    gst::init()?;

    let pipeline = gst::Pipeline::default();
    let src = gst::ElementFactory::make(source).build()?;
    let appsink = gst_app::AppSink::builder()
        .caps(
            &gst_audio::AudioCapsBuilder::new_interleaved()
                .format(gst_audio::AUDIO_FORMAT_F32)
                //.format(gst_audio::AUDIO_FORMAT_S16)
                .channels(1)
                .rate(16000)
                .build(),
        )
        .build();

    pipeline.add_many(&[&src, appsink.upcast_ref()])?;
    src.link(&appsink)?;

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or_else(|| {
                    gst::element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to get buffer from appsink")
                    );
                    gst::FlowError::Error
                })?;
                let map = buffer.map_readable().map_err(|_| {
                    gst::element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to map buffer readable")
                    );

                    gst::FlowError::Error
                })?;

                let samples = map.as_slice_of::<f32>().map_err(|_| {
                    //let samples = map.as_slice_of::<i16>().map_err(|_| {
                    gst::element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to interprete buffer as F32 PCM") //("Failed to interprete buffer as S16 PCM")
                    );

                    gst::FlowError::Error
                })?;

                let mut buf = buf.lock().unwrap();
                buf.extend(samples);
                let buf_len = buf.len();
                if buf_len > buf_size {
                    *buf = buf.split_off(buf_len - buf_size);
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}

fn whisper(
    ctx: &mut WhisperContext,
    language: &Option<String>,
    audio_data: &[f32],
    result_sender: &glib::Sender<(String, bool)>,
    fix: bool,
) {
    let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });
    //params.set_n_threads(4);
    params.set_translate(false);
    if let Some(language) = language {
        params.set_language(language);
    }
    params.set_no_context(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_single_segment(true);
    // experimental
    //params.set_speed_up(true);

    ctx.full(params, &audio_data[..])
        .expect("Failed to run model");

    let num_segments = ctx.full_n_segments();
    debug!("{}", audio_data.len());
    let mut result = String::new();
    for i in 0..num_segments {
        let segment = ctx.full_get_segment_text(i).expect("Failed to get segment");
        result.push_str(&segment);
    }
    debug!("{result}");
    result_sender.send((result, fix)).unwrap();
}

#[derive(Clone)]
struct Window {
    window: adw::ApplicationWindow,
    scrolled: gtk::ScrolledWindow,
    vbox: gtk::Box,
    scrollbacks: VecDeque<gtk::Label>,
    label: gtk::Label,
    font_size: i32,
}

const MAX_SCROLLBACKS: usize = 100;

impl Window {
    fn new(app: &adw::Application, args: &Args) -> Self {
        let win = adw::ApplicationWindow::new(app);
        win.set_size_request(200, 16);
        win.set_default_size(args.width, args.height);
        win.set_title(Some(env!("CARGO_PKG_NAME")));

        let scrolled = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .build();
        win.set_content(Some(&scrolled));

        let vbox = gtk::Box::builder()
            .orientation(gtk::Orientation::Vertical)
            .halign(gtk::Align::Start)
            .build();
        scrolled.set_child(Some(&vbox));

        let label = Self::new_label(args.font_size);
        vbox.append(&label);

        // make window draggable
        let gesture = gtk::GestureDrag::new();
        {
            let win = win.clone();
            gesture.connect_drag_update(move |gesture, _offset_x, _offset_y| {
                gesture.set_state(gtk::EventSequenceState::Claimed);
                if let Some((start_x, start_y)) = gesture.start_point() {
                    let native = win.native().unwrap();
                    let (mut window_x, mut window_y) = win
                        .translate_coordinates(&native, start_x, start_y)
                        .unwrap();
                    let (native_x, native_y) = native.surface_transform();
                    window_x += native_x;
                    window_y += native_y;
                    let surface = win.surface();
                    if let Some(toplevel) = surface.downcast_ref::<gdk::Toplevel>() {
                        toplevel.begin_move(
                            &gesture.device().unwrap(),
                            gesture.current_button() as i32,
                            window_x,
                            window_y,
                            gesture.current_event_time(),
                        );
                    }
                    gesture.reset();
                }
            });
        }
        win.add_controller(&gesture);

        Window {
            window: win,
            scrolled,
            vbox,
            scrollbacks: VecDeque::new(),
            label,
            font_size: args.font_size,
        }
    }

    fn new_label(font_size: i32) -> gtk::Label {
        let attrs = pango::AttrList::new();
        attrs.insert(pango::AttrSize::new(font_size * pango::SCALE));
        let label = gtk::Label::builder()
            .wrap(true)
            .wrap_mode(pango::WrapMode::WordChar)
            .natural_wrap_mode(gtk::NaturalWrapMode::None)
            .halign(gtk::Align::Start)
            .justify(gtk::Justification::Left)
            .attributes(&attrs)
            .build();

        label
    }

    fn fix_label(&mut self) {
        self.scrollbacks.push_back(self.label.clone());
        while self.scrollbacks.len() > MAX_SCROLLBACKS {
            if let Some(label) = self.scrollbacks.pop_front() {
                self.vbox.remove(&label);
            }
        }
        let label = Self::new_label(self.font_size);
        self.vbox.append(&label);
        self.label = label;
    }
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    stderrlog::new()
        .module(module_path!())
        .timestamp(stderrlog::Timestamp::Second)
        .verbosity(args.verbose as usize)
        .init()
        .unwrap();

    let main_loop = glib::MainLoop::new(None, false);

    let audio_buf_size = 16000 * args.length / 1000;
    let max_buf_size = audio_buf_size * 2;
    let buf = Arc::new(Mutex::new(VecDeque::new()));
    let pipeline = create_pipeline(&args.source, buf.clone(), max_buf_size).unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();

    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    {
        let main_loop = main_loop.clone();
        bus.add_watch(move |_, msg| {
            use gst::MessageView;
            match msg.view() {
                MessageView::Eos(..) => main_loop.quit(),
                MessageView::Error(err) => {
                    error!(
                        "Erro from {:?}: {} ({:?})",
                        err.src().map(|s| s.path_string()),
                        err.error(),
                        err.debug(),
                    );
                    main_loop.quit();
                }
                _ => (),
            };

            glib::Continue(true)
        })
        .expect("Failed to add bus watch");
    }

    let app = adw::Application::new(
        Some(&format!("org.u7fa9.{}", env!("CARGO_PKG_NAME"))),
        gio::ApplicationFlags::FLAGS_NONE,
    );
    gio::resources_register_include!("styles.gresource").expect("Failed to register resources.");
    app.connect_activate(move |app| {
        let (result_sender, result_receiver) =
            glib::MainContext::channel(glib::source::PRIORITY_DEFAULT);

        let (tick_sender, tick_receiver) = sync_channel::<()>(1);
        let mut ctx = WhisperContext::new(&args.model)
            .expect(&format!("Failed to load model {}", &args.model));
        glib::timeout_add(Duration::from_millis(args.step), move || {
            let _ = tick_sender.try_send(());
            glib::Continue(true)
        });
        let buf = buf.clone();
        let language = args.language.clone();
        thread::spawn(move || {
            let mut fix_next = false;
            loop {
                tick_receiver.recv().unwrap();
                let fix = fix_next;
                let mut audio_data = Vec::new();
                {
                    let mut buf = buf.lock().unwrap();
                    buf.make_contiguous().clone_into(&mut audio_data);
                    if buf.len() >= audio_buf_size {
                        buf.clear();
                        let keep_size = 16000 * args.keep / 1000;
                        buf.extend(&audio_data[(audio_data.len() - keep_size)..]);
                        fix_next = true;
                    } else {
                        fix_next = false;
                    }
                }
                whisper(&mut ctx, &language, &audio_data, &result_sender, fix);
            }
        });

        let win = Window::new(&app, &args);

        result_receiver.attach(None, {
            let mut win = win.clone();
            move |(text, fix)| {
                if fix {
                    win.fix_label();
                }
                win.label.set_text(&text);

                // scroll window to bottom if last two of labels are displayed
                // (don't scroll if you read scrollbacks)
                glib::idle_add_local_once({
                    let scrolled = win.scrolled.clone();
                    let check_label = win.scrollbacks.back().unwrap_or(&win.label).clone();
                    move || {
                        let vadj = scrolled.vadjustment();
                        if let Some((_, y)) = check_label.translate_coordinates(&scrolled, 0.0, 0.0)
                        {
                            if y <= vadj.page_size() {
                                scrolled.emit_scroll_child(gtk::ScrollType::End, false);
                            }
                        }
                    }
                });

                glib::Continue(true)
            }
        });

        win.window.show();
    });

    //main_loop.run();
    let args: &[String] = &[];
    app.run_with_args(args);

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}
