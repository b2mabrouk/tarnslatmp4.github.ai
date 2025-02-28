document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    const uploadForm = document.getElementById('upload-form');
    const youtubeForm = document.getElementById('youtube-form');
    const loadingIndicator = document.querySelector('.loading-indicator');
    const progressInfo = document.querySelector('.progress-info');
    const resultContent = document.querySelector('.result-content');
    const srtPreview = document.querySelector('.srt-preview');
    const downloadBtn = document.querySelector('.download-btn');
    
    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Handle file upload form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('video-file');
        const languageSelect = document.getElementById('language-select-file');
        
        if (!fileInput.files.length) {
            alert('الرجاء اختيار ملف فيديو');
            return;
        }
        
        const file = fileInput.files[0];
        console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', file.type);
        
        if (file.size > 100 * 1024 * 1024) {  // 100MB limit
            alert('حجم الملف كبير جدًا. الحد الأقصى هو 100 ميجابايت.');
            return;
        }
        
        // Check if file is a video
        if (!file.type.startsWith('video/')) {
            alert('الرجاء اختيار ملف فيديو صالح.');
            return;
        }
        
        const formData = new FormData();
        formData.append('video', file);
        formData.append('language', languageSelect.value);
        
        // Show loading indicator and hide other containers
        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('error-container').style.display = 'none';
        document.getElementById('result-container').style.display = 'none';
        document.getElementById('progress-info').textContent = 'جاري تحميل الفيديو...';
        document.getElementById('progress-bar').style.width = '0%';
        
        // Send request
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Upload response:', data);
            
            if (data.task_id) {
                // Start checking progress
                checkProgress(data.task_id);
            } else if (data.subtitles || data.srt_content) {
                // Direct response with subtitles
                displayResult(data);
            } else {
                throw new Error('استجابة غير صالحة من الخادم');
            }
        })
        .catch(error => handleError(error, 'حدث خطأ أثناء تحميل الفيديو'));
    });
    
    // Handle YouTube form submission
    youtubeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const youtubeUrl = document.getElementById('youtube-url').value;
        const languageSelect = document.getElementById('language-select-youtube');
        
        if (!youtubeUrl) {
            alert('الرجاء إدخال رابط فيديو يوتيوب');
            return;
        }
        
        // Validate YouTube URL
        if (!youtubeUrl.match(/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/)) {
            alert('الرجاء إدخال رابط يوتيوب صالح');
            return;
        }
        
        // Show loading indicator and hide other containers
        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('error-container').style.display = 'none';
        document.getElementById('result-container').style.display = 'none';
        document.getElementById('progress-info').textContent = 'جاري تحميل الفيديو من يوتيوب...';
        document.getElementById('progress-bar').style.width = '0%';
        
        // Send request to the correct endpoint
        fetch('/process_youtube', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                youtube_url: youtubeUrl,
                language: languageSelect.value
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('YouTube response:', data);
            
            if (data.task_id) {
                // Start checking progress
                checkProgress(data.task_id);
            } else if (data.subtitles || data.srt_content) {
                // Direct response with subtitles
                displayResult(data);
            } else {
                throw new Error('استجابة غير صالحة من الخادم');
            }
        })
        .catch(error => handleError(error, 'حدث خطأ أثناء تحميل الفيديو من يوتيوب'));
    });
    
    // Add CSS to improve visibility of result container
    document.head.insertAdjacentHTML('beforeend', `
<style>
#result-container {
    margin-top: 20px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #f9f9f9;
}

.srt-preview {
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: monospace;
    padding: 10px;
    border: 1px solid #ccc;
    background-color: white;
    direction: ltr;
    text-align: left;
    margin-bottom: 15px;
}

.download-btn {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    display: block;
    margin: 10px auto;
}

.download-btn:hover {
    background-color: #45a049;
}

.warning {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 10px;
}
</style>
`);
    
    function displayResult(data) {
        console.log('Displaying result:', data);
        
        // Hide progress container
        document.getElementById('progress-container').style.display = 'none';
        
        // Show result container
        document.getElementById('result-container').style.display = 'block';
        
        // Get references to elements
        const srtPreview = document.querySelector('.srt-preview');
        const downloadBtn = document.querySelector('.download-btn');
        
        // Display warning if any
        if (data.warning) {
            const warningDiv = document.createElement('div');
            warningDiv.className = 'warning';
            warningDiv.textContent = data.warning;
            document.getElementById('result-container').prepend(warningDiv);
        }
        
        // Display SRT preview
        const content = data.srt_content || data.subtitles;
        if (content) {
            srtPreview.textContent = content;
            console.log('SRT content length:', content.length);
        } else {
            console.error('No SRT content found in data:', data);
            srtPreview.textContent = 'لم يتم العثور على محتوى الترجمة';
        }
        
        // Setup download button
        downloadBtn.onclick = function() {
            const content = data.srt_content || data.subtitles;
            if (!content) {
                alert('لا يوجد محتوى للتنزيل');
                return;
            }
            
            const blob = new Blob([content], { type: 'text/srt' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = data.filename || 'subtitles.srt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };
    }
    
    // Function to check progress of a task
    function checkProgress(taskId) {
        if (!taskId) {
            console.error('No task ID provided');
            return;
        }
        
        console.log(`Checking progress for task: ${taskId}`);
        
        fetch(`/progress/${taskId}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            },
            cache: 'no-store'
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Progress data:', data);
                
                if (data.status === 'completed') {
                    console.log('Task completed with result:', data.result);
                    if (data.result && (data.result.srt_content || data.result.subtitles)) {
                        displayResult(data.result);
                    } else {
                        console.error('No result or content in completed task:', data);
                        handleError(new Error('لم يتم العثور على محتوى الترجمة'), 'اكتملت المعالجة ولكن لم يتم العثور على محتوى الترجمة');
                    }
                } else if (data.status === 'error') {
                    handleError(new Error(data.error || data.message || 'حدث خطأ غير معروف'), 'حدث خطأ أثناء معالجة الفيديو');
                } else {
                    // Update progress
                    const progressPercent = data.progress || 0;
                    const progressBar = document.getElementById('progress-bar');
                    progressBar.style.width = `${progressPercent}%`;
                    document.getElementById('progress-info').textContent = `${data.message || 'جاري المعالجة...'} (${progressPercent}%)`;
                    
                    // Continue checking progress
                    setTimeout(() => checkProgress(taskId), 1000);
                }
            })
            .catch(error => handleError(error, 'حدث خطأ أثناء التحقق من التقدم'));
    }
    
    function handleError(error, message = 'حدث خطأ أثناء المعالجة') {
        console.error(error);
        document.getElementById('progress-container').style.display = 'none';
        document.getElementById('error-container').style.display = 'block';
        document.getElementById('error-message').textContent = message;
        
        // Log detailed error information
        if (error && error.message) {
            console.error('Error details:', error.message);
        }
        
        // Reset the form
        document.getElementById('upload-form').reset();
    }
});
