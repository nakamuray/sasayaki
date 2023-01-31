fn main() {
    glib_build_tools::compile_resources(
        "src/resources",
        "src/resources/styles.gresource.xml",
        "styles.gresource",
    );
}
