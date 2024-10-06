### How to do profiling???

command example: ```ncu --set full sum```


### Why the time of execution of GPU is higher than CPU when i profile with Nsight profile instead of gettimeofday()

- **gettimeofday():** Measures the execution time from a higher-level perspective, focusing on how long the entire operation takes without profiling overhead.
- **Nsight Profiler:** Measures kernel performance, including overhead for collecting metrics, resulting in longer times.


### For details on my GPU

command: ``` nvidia-smi -q -i 0 ```

```sh
==============NVSMI LOG==============

Timestamp                                 : Mon Sep 30 09:11:41 2024
Driver Version                            : 535.183.01
CUDA Version                              : 12.2

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GeForce RTX 3050
    Product Brand                         : GeForce
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Enabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : N/A
        Pending                           : N/A
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : N/A
    GPU UUID                              : GPU-d562b328-39e4-284b-0ee7-fa723c2e3195
    Minor Number                          : 0
    VBIOS Version                         : 94.07.68.00.AA
    MultiGPU Board                        : No
    Board ID                              : 0x100
    Board Part Number                     : N/A
    GPU Part Number                       : 2582-350-A1
    FRU Part Number                       : N/A
    Module ID                             : 1
    Inforom Version
        Image Version                     : G001.0000.94.01
        OEM Object                        : 2.0
        ECC Object                        : N/A
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : N/A
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x01
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x258210DE
        Bus Id                            : 00000000:01:00.0
        Sub System Id                     : 0x8D9D1462
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 5
            Link Width
                Max                       : 16x
                Current                   : 8x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 7000 KB/s
        Rx Throughput                     : 39000 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : 0 %
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    Sparse Operation Mode                 : N/A
    FB Memory Usage
        Total                             : 8192 MiB
        Reserved                          : 226 MiB
        Used                              : 329 MiB
        Free                              : 7636 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 14 MiB
        Free                              : 242 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 9 %
        Memory                            : 3 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : N/A
        Pending                           : N/A
    ECC Errors
        Volatile
            SRAM Correctable              : N/A
            SRAM Uncorrectable Parity     : N/A
            SRAM Uncorrectable SEC-DED    : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
        Aggregate
            SRAM Correctable              : N/A
            SRAM Uncorrectable Parity     : N/A
            SRAM Uncorrectable SEC-DED    : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
            SRAM Threshold Exceeded       : N/A
        Aggregate Uncorrectable SRAM Sources
            SRAM L2                       : N/A
            SRAM SM                       : N/A
            SRAM Microcontroller          : N/A
            SRAM PCIE                     : N/A
            SRAM Other                    : N/A
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows                         : N/A
    Temperature
        GPU Current Temp                  : 47 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 97 C
        GPU Slowdown Temp                 : 94 C
        GPU Max Operating Temp            : 92 C
        GPU Target Temperature            : 83 C
        Memory Current Temp               : N/A
        Memory Max Operating Temp         : N/A
    GPU Power Readings
        Power Draw                        : N/A
        Current Power Limit               : 115.00 W
        Requested Power Limit             : 115.00 W
        Default Power Limit               : 115.00 W
        Min Power Limit                   : 50.00 W
        Max Power Limit                   : 115.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 1807 MHz
        SM                                : 1807 MHz
        Memory                            : 7000 MHz
        Video                             : 1590 MHz
    Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Default Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 2250 MHz
        SM                                : 2250 MHz
        Memory                            : 7001 MHz
        Video                             : 1950 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 931.250 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1700
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 47 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 198295
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 159 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 198474
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 31 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 200034
            Type                          : G
            Name                          : /opt/google/chrome/chrome --type=gpu-process --string-annotations --crashpad-handler-pid=199999 --enable-crash-reporter=e7603d11-3d05-4717-8976-f6f2892ab9cb, --no-subproc-heap-profiling --change-stack-guard-on-fork=enable --gpu-preferences=UAAAAAAAAAAgAAAEAAAAAAAAAAAAAAAAAABgAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAABAAAAAAAAAAEAAAAAAAAAAIAAAAAAAAAAgAAAAAAAAA --shared-files --field-trial-handle=3,i,15573719832756964331,12545793221491190720,262144 --variations-seed-version=20240927-160657.442000
            Used GPU Memory               : 23 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 206935
            Type                          : G
            Name                          : /snap/code/169/usr/share/code/code --type=gpu-process --disable-gpu-sandbox --no-sandbox --crashpad-handler-pid=206923 --enable-crash-reporter=6a7b0502-6294-4a58-923c-84b4ad5f1325,no_channel --user-data-dir=/home/lion/.config/Code --gpu-preferences=WAAAAAAAAAAgAAAEAAAAAAAAAAAAAAAAAABgAAEAAAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAGAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA== --shared-files --field-trial-handle=3,i,4197171861039077421,358576750334003591,262144 --enable-features=kWebSQLAccess --disable-features=CalculateNativeWinOcclusion,SpareRendererForSitePerProcess --variations-seed-version
            Used GPU Memory               : 49 MiB


```