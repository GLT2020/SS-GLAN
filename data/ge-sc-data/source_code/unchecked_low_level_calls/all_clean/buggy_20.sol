pragma solidity ^0.5.10;
contract Ownable {
  function bug_unchk_send29() payable public{
      msg.sender.transfer(1 ether);}
  address public owner;
  function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
  event OwnerChanged(address oldOwner, address newOwner);
    constructor() internal {
        owner = msg.sender;
    }
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner() {
        require(msg.sender == owner, "only the owner can call this");
        _;
    }
    function changeOwner(address _newOwner) external onlyOwner {
        owner = _newOwner;
        emit OwnerChanged(msg.sender, _newOwner);
    }
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
}
contract Stoppable is Ownable {
  function bug_unchk_send6() payable public{
      msg.sender.transfer(1 ether);}
  bool public isActive = true;
  function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
  event IsActiveChanged(bool _isActive);
    modifier onlyActive() {
        require(isActive, "contract is stopped");
        _;
    }
    function setIsActive(bool _isActive) external onlyOwner {
        if (_isActive == isActive) return;
        isActive = _isActive;
        emit IsActiveChanged(_isActive);
    }
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
}
contract RampInstantPoolInterface {
    uint16 public ASSET_TYPE;
    function sendFundsToSwap(uint256 _amount)
        public  returns(bool success);
}
contract RampInstantEscrowsPoolInterface {
    uint16 public ASSET_TYPE;
    function release(
        address _pool,
        address payable _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    )
        external;
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);} 
    function returnFunds(
        address payable _pool,
        address _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    )
        external;
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);} 
}
contract RampInstantPool is Ownable, Stoppable, RampInstantPoolInterface {
    uint256 constant private MAX_SWAP_AMOUNT_LIMIT = 1 << 240;
    uint16 public ASSET_TYPE;
  function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
  address payable public swapsContract;
  function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public minSwapAmount;
  function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public maxSwapAmount;
  function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
  bytes32 public paymentDetailsHash;
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event ReceivedFunds(address _from, uint256 _amount);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event LimitsChanged(uint256 _minAmount, uint256 _maxAmount);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event SwapsContractChanged(address _oldAddress, address _newAddress);
    constructor(
        address payable _swapsContract,
        uint256 _minSwapAmount,
        uint256 _maxSwapAmount,
        bytes32 _paymentDetailsHash,
        uint16 _assetType
    )
        public
        validateLimits(_minSwapAmount, _maxSwapAmount)
        validateSwapsContract(_swapsContract, _assetType)
    {
        swapsContract = _swapsContract;
        paymentDetailsHash = _paymentDetailsHash;
        minSwapAmount = _minSwapAmount;
        maxSwapAmount = _maxSwapAmount;
        ASSET_TYPE = _assetType;
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function availableFunds() public view returns (uint256);
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function withdrawFunds(address payable _to, uint256 _amount)
        public  returns (bool success);
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function withdrawAllFunds(address payable _to) public onlyOwner returns (bool success) {
        return withdrawFunds(_to, availableFunds());
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function setLimits(
        uint256 _minAmount,
        uint256 _maxAmount
    ) public onlyOwner validateLimits(_minAmount, _maxAmount) {
        minSwapAmount = _minAmount;
        maxSwapAmount = _maxAmount;
        emit LimitsChanged(_minAmount, _maxAmount);
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function setSwapsContract(
        address payable _swapsContract
    ) public onlyOwner validateSwapsContract(_swapsContract, ASSET_TYPE) {
        address oldSwapsContract = swapsContract;
        swapsContract = _swapsContract;
        emit SwapsContractChanged(oldSwapsContract, _swapsContract);
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function sendFundsToSwap(uint256 _amount)
        public  returns(bool success);
    function releaseSwap(
        address payable _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    ) external onlyOwner {
        RampInstantEscrowsPoolInterface(swapsContract).release(
            address(this),
            _receiver,
            _oracle,
            _assetData,
            _paymentDetailsHash
        );
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function returnSwap(
        address _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    ) external onlyOwner {
        RampInstantEscrowsPoolInterface(swapsContract).returnFunds(
            address(this),
            _receiver,
            _oracle,
            _assetData,
            _paymentDetailsHash
        );
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function () external payable {
        revert("this pool cannot receive ether");
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlySwapsContract() {
        require(msg.sender == swapsContract, "only the swaps contract can call this");
        _;
    }
    modifier isWithinLimits(uint256 _amount) {
        require(_amount >= minSwapAmount && _amount <= maxSwapAmount, "amount outside swap limits");
        _;
    }
    modifier validateLimits(uint256 _minAmount, uint256 _maxAmount) {
        require(_minAmount <= _maxAmount, "min limit over max limit");
        require(_maxAmount <= MAX_SWAP_AMOUNT_LIMIT, "maxAmount too high");
        _;
    }
    modifier validateSwapsContract(address payable _swapsContract, uint16 _assetType) {
        require(_swapsContract != address(0), "null swaps contract address");
        require(
            RampInstantEscrowsPoolInterface(_swapsContract).ASSET_TYPE() == _assetType,
            "pool asset type doesn't match swap contract"
        );
        _;
    }
}
contract RampInstantEthPool is RampInstantPool {
  function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
  uint16 internal constant ETH_TYPE_ID = 1;
    constructor(
        address payable _swapsContract,
        uint256 _minSwapAmount,
        uint256 _maxSwapAmount,
        bytes32 _paymentDetailsHash
    )
        public
        RampInstantPool(
            _swapsContract, _minSwapAmount, _maxSwapAmount, _paymentDetailsHash, ETH_TYPE_ID
        )
    {}
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function availableFunds() public view returns(uint256) {
        return address(this).balance;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function withdrawFunds(
        address payable _to,
        uint256 _amount
    ) public onlyOwner returns (bool success) {
        _to.transfer(_amount);  
        return true;
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function sendFundsToSwap(
        uint256 _amount
    ) public onlyActive onlySwapsContract isWithinLimits(_amount) returns(bool success) {
        swapsContract.transfer(_amount);  
        return true;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function () external payable {
        require(msg.data.length == 0, "invalid pool function called");
        if (msg.sender != swapsContract) {
            emit ReceivedFunds(msg.sender, msg.value);
        }
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
}