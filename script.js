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
        
        const formData = new FormData();
        formData.append('video', fileInput.files[0]);
        formData.append('language', languageSelect.value);
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultContent.style.display = 'none';
        progressInfo.textContent = 'جاري تحميل الفيديو...';
        
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
        .catch(error => {
            console.error('Error uploading file:', error);
            loadingIndicator.style.display = 'none';
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = `خطأ: ${error.message}`;
            resultContent.innerHTML = '';
            resultContent.appendChild(errorDiv);
            resultContent.style.display = 'block';
        });
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
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultContent.style.display = 'none';
        progressInfo.textContent = 'جاري تحميل الفيديو من يوتيوب...';
        
        // Send request
        fetch('/process', {
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
        .catch(error => {
            console.error('Error processing YouTube video:', error);
            loadingIndicator.style.display = 'none';
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = `خطأ: ${error.message}`;
            resultContent.innerHTML = '';
            resultContent.appendChild(errorDiv);
            resultContent.style.display = 'block';
        });
    });
    
    function displayResult(data) {
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        progressInfo.textContent = '';
        
        // Show result content
        resultContent.style.display = 'block';
        
        // Display warning if any
        if (data.warning) {
            const warningDiv = document.createElement('div');
            warningDiv.className = 'warning';
            warningDiv.textContent = data.warning;
            resultContent.prepend(warningDiv);
        }
        
        // Display SRT preview
        const content = data.srt_content || data.subtitles;
        srtPreview.textContent = content;
        
        // Setup download button
        downloadBtn.onclick = function() {
            const content = data.srt_content || data.subtitles;
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
        
        fetch(`/progress/${taskId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Progress data:', data);
                
                if (data.status === 'completed') {
                    console.log('Task completed');
                    displayResult(data.result);
                } else if (data.status === 'error') {
                    console.error('Task error:', data.error);
                    loadingIndicator.style.display = 'none';
                    progressInfo.textContent = '';
                    
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error';
                    errorDiv.textContent = `خطأ: ${data.error || data.message || 'حدث خطأ غير معروف'}`;
                    resultContent.prepend(errorDiv);
                    resultContent.style.display = 'block';
                } else {
                    // Update progress
                    const progressPercent = data.progress || 0;
                    progressInfo.textContent = `${data.message || 'جاري المعالجة...'} (${progressPercent}%)`;
                    
                    // Continue checking progress
                    setTimeout(() => checkProgress(taskId), 1000);
                }
            })
            .catch(error => {
                console.error('Error checking progress:', error);
                loadingIndicator.style.display = 'none';
                progressInfo.textContent = '';
                
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error';
                errorDiv.textContent = `خطأ في التحقق من التقدم: ${error.message}`;
                resultContent.prepend(errorDiv);
                resultContent.style.display = 'block';
            });
    }
});
